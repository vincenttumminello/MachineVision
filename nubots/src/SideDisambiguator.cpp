#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <opencv2/core.hpp>
#include "SideDisambiguator.h"

SideDisambiguator::SideDisambiguator(const FisheyeLens & lens, const FieldDimensions & dims, const Options & opts)
    : options(opts)
    , lens_(lens)
    , detector_(lens, dims)
    , halfCarpetLength_(dims.fieldLength/2 + dims.borderStripMinWidth + opts.fieldMargin)
    , halfCarpetWidth_(dims.fieldWidth/2 + dims.borderStripMinWidth + opts.fieldMargin)
{}

SideDisambiguator::SideDisambiguator(const FisheyeLens & lens, const FieldDimensions & dims)
    : SideDisambiguator(lens, dims, Options{})
{}

// Orthonormal tangent basis of a unit vector (columns span the plane normal to u).
static Eigen::Matrix<double, 3, 2> tangentBasis(const Eigen::Vector3d & u)
{
    Eigen::Vector3d t1 = u.cross(Eigen::Vector3d::UnitZ());
    if (t1.squaredNorm() < 1e-8)
    {
        t1 = u.cross(Eigen::Vector3d::UnitX());
    }
    t1.normalize();
    Eigen::Matrix<double, 3, 2> T;
    T.col(0) = t1;
    T.col(1) = u.cross(t1).normalized();
    return T;
}

bool SideDisambiguator::isBackgroundPoint(const Eigen::Vector3d & rPFf) const
{
    // Plausible static background: beyond the carpet in plan view, or overhead
    // structure (ceiling lights/trusses above the field are static and useful).
    // Reject underground or absurdly high solutions outright.
    if (rPFf.z() < -1.0 || rPFf.z() > 15.0)
    {
        return false;
    }
    const bool offCarpet = std::abs(rPFf.x()) > halfCarpetLength_ || std::abs(rPFf.y()) > halfCarpetWidth_;
    return offCarpet || rPFf.z() > options.minHeightOnCarpet;
}

SideDisambiguator::TriResult SideDisambiguator::triangulate(const std::vector<Landmark::Obs> & obs,
                                                            Eigen::Vector3d & rPFf, Eigen::Matrix3d & P, double & meanChi2) const
{
    if (obs.size() < 2)
    {
        return TriResult::GEOMETRY;
    }

    // Linear least squares for the point minimising the perpendicular distance
    // to every observation ray: sum_i || (I - u_i u_i^T)(p - c_i) ||^2.
    Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();
    for (const Landmark::Obs & o : obs)
    {
        const Eigen::Matrix3d M = Eigen::Matrix3d::Identity() - o.uFf*o.uFf.transpose();
        A += M;
        b += M*o.rCFf;
    }

    // Depth is unobservable without parallax: the normal matrix is singular
    // along the (common) ray direction.
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(A);
    if (es.info() != Eigen::Success || es.eigenvalues()(0) < 1e-3)
    {
        return TriResult::GEOMETRY;
    }
    rPFf = A.ldlt().solve(b);

    // Static-consistency test: every observation must point at the solved point
    // to within the angular residual scale. Dynamic objects (crowd, robots)
    // cannot satisfy this once there is parallax in the window.
    const double sigma2 = options.sigmaStatic*options.sigmaStatic;
    double chi2 = 0.0;
    Eigen::Matrix3d info = Eigen::Matrix3d::Identity()*(1.0/(options.maxRange*options.maxRange));
    for (const Landmark::Obs & o : obs)
    {
        const Eigen::Vector3d rel = rPFf - o.rCFf;
        const double range = rel.norm();
        const double depth = rel.dot(o.uFf);
        if (depth < options.minRange || range > options.maxRange)
        {
            return TriResult::RANGE;   // Behind or implausibly close/far for a background point
        }
        const Eigen::Matrix3d M = Eigen::Matrix3d::Identity() - o.uFf*o.uFf.transpose();
        chi2 += (M*rel).squaredNorm()/(sigma2*range*range);
        info += M/(sigma2*range*range);
    }
    meanChi2 = chi2/obs.size();
    if (meanChi2 > options.staticChi2Mean)
    {
        return TriResult::CHI2;
    }
    P = info.inverse();
    return TriResult::OK;
}

bool SideDisambiguator::fitFar(const std::vector<Landmark::Obs> & obs,
                               Eigen::Vector3d & rPFf, Eigen::Matrix3d & P) const
{
    Eigen::Vector3d uSum = Eigen::Vector3d::Zero();
    Eigen::Vector3d cMean = Eigen::Vector3d::Zero();
    for (const Landmark::Obs & o : obs)
    {
        uSum += o.uFf;
        cMean += o.rCFf;
    }
    if (uSum.norm() < 1e-9)
    {
        return false;
    }
    const Eigen::Vector3d uMean = uSum.normalized();
    cMean /= obs.size();

    double spread2 = 0.0;
    for (const Landmark::Obs & o : obs)
    {
        const double a = std::acos(std::clamp(o.uFf.dot(uMean), -1.0, 1.0));
        spread2 += a*a;
    }
    const double spreadRms = std::sqrt(spread2/obs.size());
    if (spreadRms > options.farMaxSpread)
    {
        return false;   // Jittery bearings: dynamic object or track hopping
    }

    rPFf = cMean + options.assumedRange*uMean;
    const double sigmaR = 0.5*options.assumedRange;                                     // Depth stand-in
    const double sigmaPerp = std::max(spreadRms, options.sigmaStatic)*options.assumedRange;
    P = sigmaR*sigmaR*uMean*uMean.transpose()
      + sigmaPerp*sigmaPerp*(Eigen::Matrix3d::Identity() - uMean*uMean.transpose());
    return true;
}

std::vector<SideDisambiguator::Association> SideDisambiguator::associate(
    const std::vector<OutOfFieldFeature> & features, const Pose<double> & Tfc,
    double posStd, double yawStd,
    double & score, std::size_t & nPredictedVisible, std::vector<std::size_t> & visibleMiss) const
{
    score = 0.0;
    nPredictedVisible = 0;
    visibleMiss.clear();

    const Eigen::Matrix3d Rcf = Tfc.rotationMatrix.transpose();
    const Eigen::Vector3d rCFf = Tfc.translationVector;
    const double logClutter = std::log(options.clutterDensity);
    const double cosPreGate = std::cos(options.preGateAngle);
    const double effPosStd = std::max(posStd, options.posStdFloor);
    const Eigen::Matrix3d Pcam = Eigen::Matrix3d::Identity()*effPosStd*effPosStd;

    // Predict every landmark into this camera and precompute its predictive
    // density in the ray tangent plane.
    struct Predicted
    {
        std::size_t idx;                    ///< Index into landmarks_
        Eigen::Vector3d uFf;                ///< Predicted unit ray in {f}
        Eigen::Matrix<double, 3, 2> T;      ///< Tangent basis at the predicted ray
        Eigen::Matrix2d Sinv;               ///< Inverse innovation covariance
        double halfLogDet2piS;              ///< 0.5*log det(2 pi S)
        bool wellInside;                    ///< Predicted comfortably inside the image
    };
    std::vector<Predicted> predicted;
    predicted.reserve(landmarks_.size());
    for (std::size_t j = 0; j < landmarks_.size(); ++j)
    {
        const Landmark & lm = landmarks_[j];
        const Eigen::Vector3d rel = lm.rPFf - rCFf;
        const double range = rel.norm();
        if (range < options.minRange) continue;

        const Eigen::Vector3d uFf = rel/range;
        const Eigen::Vector3d uCc = Rcf*uFf;
        if (!FisheyeLens::inFrontOfCamera(uCc)) continue;
        const Eigen::Vector2d px = lens_.project(uCc);
        if (!lens_.inImage(px)) continue;

        Predicted pr;
        pr.idx = j;
        pr.uFf = uFf;
        pr.T = tangentBasis(uFf);
        const double m = options.visibleMargin;
        pr.wellInside = px.x() >= m && px.x() < lens_.width - m && px.y() >= m && px.y() < lens_.height - m;

        // Innovation covariance in the tangent plane: bearing noise + landmark
        // and camera position uncertainty projected across the range + camera
        // yaw uncertainty (rotation about field-up).
        const Eigen::Vector2d a = pr.T.transpose()*Eigen::Vector3d::UnitZ().cross(uFf);
        Eigen::Matrix2d S = Eigen::Matrix2d::Identity()*(options.sigmaAngular*options.sigmaAngular)
                          + pr.T.transpose()*(lm.P + Pcam)*pr.T/(range*range)
                          + yawStd*yawStd*a*a.transpose();

        // A bearing-only landmark viewed far from its anchor smears into a long
        // thin acceptance corridor (its radial depth variance leaks into the
        // tangent plane). Such a prediction cannot discriminate anything at
        // this baseline: skip the landmark entirely rather than let corridor
        // matches alias, and don't count it as predicted-visible either.
        const double trS = S.trace();
        const double dS = std::sqrt(std::max(0.25*trS*trS - S.determinant(), 0.0));
        if (0.5*trS + dS > options.maxTangentSigma*options.maxTangentSigma)
        {
            continue;
        }
        const double detS = S.determinant();
        pr.Sinv = S.inverse();
        pr.halfLogDet2piS = 0.5*std::log(detS) + std::log(2.0*M_PI);
        predicted.push_back(pr);
        if (pr.wellInside) nPredictedVisible++;
    }

    // Surprisal of every gated (feature, landmark) pair. The clutter hypothesis
    // (uniform density over direction space) sets the acceptance threshold: an
    // association is only worth making if the predictive density at the residual
    // beats the clutter density.
    struct Pair { double surprisal; std::size_t f; std::size_t p; };
    std::vector<Pair> pairs;
    for (std::size_t i = 0; i < features.size(); ++i)
    {
        if (!features[i].outOfField) continue;
        const Eigen::Vector3d uMeasF = Tfc.rotationMatrix*features[i].uPCc;
        for (std::size_t k = 0; k < predicted.size(); ++k)
        {
            const Predicted & pr = predicted[k];
            if (uMeasF.dot(pr.uFf) < cosPreGate) continue;
            const int dist = static_cast<int>(cv::norm(landmarks_[pr.idx].descriptor,
                                                       features[i].descriptor, cv::NORM_HAMMING));
            if (dist > options.maxDescriptorDistance) continue;
            const Eigen::Vector2d e = pr.T.transpose()*(uMeasF - pr.uFf);
            const double s = 0.5*e.dot(pr.Sinv*e) + pr.halfLogDet2piS;
            if (s < -logClutter)
            {
                pairs.push_back({s, i, k});
            }
        }
    }

    // Greedy one-to-one assignment by ascending surprisal (SNN).
    std::sort(pairs.begin(), pairs.end(), [](const Pair & a, const Pair & b) { return a.surprisal < b.surprisal; });
    std::vector<bool> featTaken(features.size(), false);
    std::vector<bool> predTaken(predicted.size(), false);
    std::vector<Association> assoc;
    for (const Pair & pq : pairs)
    {
        if (featTaken[pq.f] || predTaken[pq.p]) continue;
        featTaken[pq.f] = true;
        predTaken[pq.p] = true;
        assoc.push_back({pq.f, predicted[pq.p].idx, pq.surprisal});
        // Robust evidence: log ratio of the inlier predictive density to the
        // clutter density (positive by the acceptance gate above).
        score += -pq.surprisal - logClutter;
    }

    for (std::size_t k = 0; k < predicted.size(); ++k)
    {
        if (predicted[k].wellInside && !predTaken[k])
        {
            visibleMiss.push_back(predicted[k].idx);
        }
    }
    return assoc;
}

void SideDisambiguator::updateCandidates(const std::vector<OutOfFieldFeature> & features,
                                         const std::vector<bool> & featureUsed,
                                         const Pose<double> & Tfc, double t)
{
    const Eigen::Vector3d rCFf = Tfc.translationVector;
    const double cosGate = std::cos(options.candGateAngle);

    // Gated (feature, candidate) pairs ranked by descriptor distance, matched
    // greedily one-to-one. The geometric gate compares against the candidate's
    // most recent bearing (no depth yet, so a generous angular gate stands in
    // for a proper prediction).
    struct Pair { int dist; std::size_t f; std::size_t c; };
    std::vector<Pair> pairs;
    for (std::size_t i = 0; i < features.size(); ++i)
    {
        if (!features[i].outOfField || featureUsed[i]) continue;
        const Eigen::Vector3d uMeasF = Tfc.rotationMatrix*features[i].uPCc;
        for (std::size_t c = 0; c < candidates_.size(); ++c)
        {
            if (uMeasF.dot(candidates_[c].obs.back().uFf) < cosGate) continue;
            const int dist = static_cast<int>(cv::norm(candidates_[c].descriptor,
                                                       features[i].descriptor, cv::NORM_HAMMING));
            if (dist > options.candMaxDescriptorDistance) continue;
            pairs.push_back({dist, i, c});
        }
    }
    std::sort(pairs.begin(), pairs.end(), [](const Pair & a, const Pair & b) { return a.dist < b.dist; });

    std::vector<bool> featMatched(features.size(), false);
    std::vector<bool> candMatched(candidates_.size(), false);
    for (const Pair & pq : pairs)
    {
        if (featMatched[pq.f] || candMatched[pq.c]) continue;
        featMatched[pq.f] = true;
        candMatched[pq.c] = true;

        Candidate & cand = candidates_[pq.c];
        const Eigen::Vector3d uMeasF = Tfc.rotationMatrix*features[pq.f].uPCc;
        if (cand.obs.size() >= static_cast<std::size_t>(options.obsWindow))
        {
            // Keep the oldest observation as the parallax/timespan anchor and
            // roll the rest of the window.
            cand.obs.erase(cand.obs.begin() + 1);
        }
        cand.obs.push_back({rCFf, uMeasF, t});
        cand.descriptor = features[pq.f].descriptor.clone();
        cand.lastSeen = t;
    }

    // Promote mature candidates: enough observations over enough time with
    // enough parallax to triangulate, and consistent with one static point.
    std::vector<bool> candDrop(candidates_.size(), false);
    for (std::size_t c = 0; c < candidates_.size(); ++c)
    {
        Candidate & cand = candidates_[c];
        if (!candMatched[c])
        {
            candDrop[c] = t - cand.lastSeen > options.candMaxAge;
            continue;
        }
        if (cand.obs.size() < static_cast<std::size_t>(options.minObs)) continue;
        if (cand.obs.back().t - cand.obs.front().t < options.minTimeSpan) continue;
        stats_.promoteAttempts++;

        double maxParallax = 0.0;
        for (std::size_t a = 0; a < cand.obs.size(); ++a)
            for (std::size_t b2 = a + 1; b2 < cand.obs.size(); ++b2)
                maxParallax = std::max(maxParallax,
                    std::acos(std::clamp(cand.obs[a].uFf.dot(cand.obs[b2].uFf), -1.0, 1.0)));
        if (maxParallax < options.minParallax)
        {
            // No usable depth. If the track is long, old and directionally
            // tight, promote it as a bearing-only landmark: bearing alone
            // discriminates the mirror. Jittery tracks are dropped as dynamic.
            if (cand.obs.size() >= static_cast<std::size_t>(options.farPromoteObs)
                && cand.obs.back().t - cand.obs.front().t >= options.farPromoteTimeSpan)
            {
                Eigen::Vector3d rPFf;
                Eigen::Matrix3d P;
                if (fitFar(cand.obs, rPFf, P) && isBackgroundPoint(rPFf))
                {
                    stats_.promotedFar++;
                    Landmark lm;
                    lm.rPFf = rPFf;
                    lm.P = P;
                    lm.far = true;
                    lm.descriptor = cand.descriptor.clone();
                    lm.hits = static_cast<int>(cand.obs.size());
                    lm.lastSeen = t;
                    lm.obs = cand.obs;
                    landmarks_.push_back(std::move(lm));
                }
                else
                {
                    stats_.farSpreadFail++;
                }
                candDrop[c] = true;
            }
            else
            {
                stats_.parallaxWait++;
            }
            continue;   // Keep waiting for baseline
        }

        Eigen::Vector3d rPFf;
        Eigen::Matrix3d P;
        double meanChi2 = 0.0;
        const TriResult tri = triangulate(cand.obs, rPFf, P, meanChi2);
        if (tri != TriResult::OK)
        {
            if (tri == TriResult::GEOMETRY) stats_.triFailGeometry++;
            else if (tri == TriResult::RANGE) stats_.triFailRange++;
            else stats_.triFailChi2++;
        }
        else if (!isBackgroundPoint(rPFf))
        {
            stats_.backgroundFail++;
        }
        else
        {
            stats_.promoted++;
            Landmark lm;
            lm.rPFf = rPFf;
            lm.P = P;
            lm.descriptor = cand.descriptor.clone();
            lm.hits = static_cast<int>(cand.obs.size());
            lm.lastSeen = t;
            lm.obs = cand.obs;
            landmarks_.push_back(std::move(lm));
        }
        // Either way the candidate is finished: promoted, or inconsistent with a
        // static background point despite sufficient parallax (dynamic object).
        candDrop[c] = true;
    }

    // Apply drops, then spawn new candidates from the remaining unmatched features.
    std::size_t w = 0;
    for (std::size_t c = 0; c < candidates_.size(); ++c)
    {
        if (candDrop[c]) continue;
        if (w != c) candidates_[w] = std::move(candidates_[c]);   // Guard the self-move
        ++w;
    }
    candidates_.resize(w);

    for (std::size_t i = 0; i < features.size(); ++i)
    {
        if (!features[i].outOfField || featureUsed[i] || featMatched[i]) continue;
        const Eigen::Vector3d uFf = Tfc.rotationMatrix*features[i].uPCc;
        // Overhead lighting grids are near-symmetric under the field's 180 deg
        // rotation; don't let high-elevation features into the map at all.
        if (std::asin(std::clamp(uFf.z(), -1.0, 1.0)) > options.maxElevation) continue;
        Candidate cand;
        cand.descriptor = features[i].descriptor.clone();
        cand.obs.push_back({rCFf, uFf, t});
        cand.lastSeen = t;
        candidates_.push_back(std::move(cand));
    }

    // Cap the candidate pool. Established tracks (more observations) outrank
    // fresh single-observation spawns, so the cap churns the spawn pool rather
    // than evicting tracks that are accumulating parallax.
    if (candidates_.size() > options.maxCandidates)
    {
        std::sort(candidates_.begin(), candidates_.end(),
                  [](const Candidate & a, const Candidate & b)
                  {
                      if (a.obs.size() != b.obs.size()) return a.obs.size() > b.obs.size();
                      return a.lastSeen > b.lastSeen;
                  });
        candidates_.resize(options.maxCandidates);
    }
}

SideDisambiguator::FrameResult SideDisambiguator::process(double t, const cv::Mat & gray,
                                                          const Pose<double> & Tfc, const Pose<double> & TfcMirror,
                                                          double posStd, double yawStd, double yawRateAbs)
{
    FrameResult res;

    res.features = detector_.detect(gray, Tfc);
    const std::vector<OutOfFieldFeature> & features = res.features;
    res.nFeatures = features.size();
    res.featureStatus.assign(features.size(), 0);
    for (std::size_t i = 0; i < features.size(); ++i)
    {
        if (features[i].outOfField)
        {
            res.featureStatus[i] = 1;
            res.nOutOfField++;
        }
    }

    // Associate the same corners against the map under both side hypotheses.
    std::size_t visOwn = 0, visMirror = 0;
    std::vector<std::size_t> missOwn, missMirror;
    std::vector<Association> assocOwn = associate(features, Tfc, posStd, yawStd,
                                                  res.scoreOwn, visOwn, missOwn);
    std::vector<Association> assocMirror = associate(features, TfcMirror, posStd, yawStd,
                                                     res.scoreMirror, visMirror, missMirror);
    res.nAssociated = assocOwn.size();
    res.nAssociatedMirror = assocMirror.size();
    res.nVisibleOwn = visOwn;
    res.nVisibleMirror = visMirror;
    for (const Association & a : assocOwn)
    {
        res.featureStatus[a.feature] = 2;
    }

    // Accumulate the side evidence whenever the map could have discriminated
    // (some landmark was predicted visible under either hypothesis) and the
    // view is trustworthy (not mid-turn: motion blur and a freshly-panned
    // viewpoint starve the true side of matches without saying anything about
    // which side is right).
    const bool scoredFrame = visOwn + visMirror > 0 && yawRateAbs < options.maxYawRate;
    if (scoredFrame)
    {
        const double delta = std::clamp(res.scoreOwn - res.scoreMirror,
                                        -options.deltaClamp, options.deltaClamp);
        llr_ = std::clamp(options.forgetting*llr_ + delta, -options.llrClamp, options.llrClamp);
    }
    res.llr = llr_;

    // Deep doubt latches: only positive evidence (not forgetting-driven decay
    // towards zero) may re-arm map building on this side.
    if (llr_ <= -options.flipThreshold) doubt_ = true;
    else if (llr_ >= options.rebuildLlr) doubt_ = false;

    // A flip needs sustained, substantive and DOMINANT mirror evidence: the LLR
    // must sit below the threshold, each streak frame must carry enough
    // mirror-side associations that a couple of matches to symmetric structure
    // cannot flip a barely-covered map, and the mirror must clearly outnumber
    // the own side (a degraded own pose is "no decision", not mirror evidence).
    // Unscored frames leave the streak untouched.
    if (scoredFrame)
    {
        if (llr_ <= -options.flipThreshold
            && res.nAssociatedMirror >= options.minFlipAssoc
            && static_cast<double>(res.nAssociatedMirror) >= options.flipDominance*static_cast<double>(res.nAssociated)
            && static_cast<double>(visOwn) >= options.flipCoverage*static_cast<double>(visMirror))
        {
            flipStreak_++;
        }
        else if (llr_ > -options.flipThreshold)
        {
            flipStreak_ = 0;    // Side no longer in doubt: clear the streak
        }
        else if (flipStreak_ > 0)
        {
            flipStreak_--;      // Still in doubt, frame unqualifying: leak, don't reset
        }
        res.flipRequested = flipStreak_ >= options.flipConsecutive;
    }

    // Map maintenance. Frozen while the pose is uncertain, the side is in
    // doubt, or a flip just happened, so a wrong-side excursion cannot poison
    // the map.
    const bool frozen = posStd > options.maxPosStd || llr_ < options.freezeLlr
                        || doubt_ || res.flipRequested || t < mapFreezeUntil_;
    res.mapFrozen = frozen;
    if (!frozen)
    {
        std::vector<bool> landmarkDead(landmarks_.size(), false);
        std::vector<bool> featureUsed(features.size(), false);

        // Hits: extend the observation window and re-triangulate. A landmark
        // that stops fitting a static point (someone who stood still and then
        // moved) fails the consistency test and is culled.
        for (const Association & a : assocOwn)
        {
            featureUsed[a.feature] = true;
            Landmark & lm = landmarks_[a.landmark];
            if (lm.obs.size() >= static_cast<std::size_t>(options.obsWindow))
            {
                // Keep the oldest observation as the parallax anchor.
                lm.obs.erase(lm.obs.begin() + 1);
            }
            lm.obs.push_back({Tfc.translationVector, Tfc.rotationMatrix*features[a.feature].uPCc, t});

            Eigen::Vector3d rPFf;
            Eigen::Matrix3d P;
            double meanChi2 = 0.0;
            const TriResult tri = triangulate(lm.obs, rPFf, P, meanChi2);
            if (tri == TriResult::OK && isBackgroundPoint(rPFf))
            {
                // Full depth solution available (possibly upgrading a
                // bearing-only landmark that finally accrued parallax).
                if (lm.far) stats_.upgraded++;
                lm.far = false;
                lm.rPFf = rPFf;
                lm.P = P;
                lm.descriptor = features[a.feature].descriptor.clone();
                lm.hits++;
                lm.missStreak = 0;
                lm.lastSeen = t;
            }
            else if (lm.far && tri != TriResult::CHI2)
            {
                // Bearing-only landmark still without parallax: refit the
                // bearing; a jittery window means a mover, so cull it.
                if (fitFar(lm.obs, rPFf, P) && isBackgroundPoint(rPFf))
                {
                    lm.rPFf = rPFf;
                    lm.P = P;
                    lm.descriptor = features[a.feature].descriptor.clone();
                    lm.hits++;
                    lm.missStreak = 0;
                    lm.lastSeen = t;
                }
                else
                {
                    landmarkDead[a.landmark] = true;
                    stats_.landmarkCulledChi2++;
                }
            }
            else
            {
                landmarkDead[a.landmark] = true;
                stats_.landmarkCulledChi2++;
            }
        }

        // Misses: predicted comfortably inside the image but not re-observed.
        for (std::size_t idx : missOwn)
        {
            if (++landmarks_[idx].missStreak > options.maxMissStreak)
            {
                landmarkDead[idx] = true;
                stats_.landmarkCulledMiss++;
            }
        }

        std::size_t w = 0;
        for (std::size_t j = 0; j < landmarks_.size(); ++j)
        {
            if (landmarkDead[j]) continue;
            if (w != j) landmarks_[w] = std::move(landmarks_[j]);   // Guard the self-move
            ++w;
        }
        landmarks_.resize(w);

        updateCandidates(features, featureUsed, Tfc, t);

        // Cap the map, keeping the most-reobserved landmarks.
        if (landmarks_.size() > options.maxLandmarks)
        {
            std::sort(landmarks_.begin(), landmarks_.end(),
                      [](const Landmark & a, const Landmark & b) { return a.hits > b.hits; });
            landmarks_.resize(options.maxLandmarks);
        }
    }

    res.nLandmarks = landmarks_.size();
    res.nCandidates = candidates_.size();
    return res;
}

void SideDisambiguator::notifyFlipApplied(double t)
{
    llr_ = -llr_;
    flipStreak_ = 0;
    mapFreezeUntil_ = t + options.flipCooldown;
}
