#include <algorithm>
#include <cmath>
#include <vector>
#include <Eigen/Core>
#include <Eigen/SVD>
#include "NeckKinematics.h"

namespace
{

/// @brief Index of the sample nearest in time to t, or npos if the stream is empty.
template <typename Sample, typename TimeOf>
std::size_t nearest(const std::vector<Sample> & v, double t, TimeOf timeOf)
{
    if (v.empty())
    {
        return std::size_t(-1);
    }
    std::size_t best = 0;
    double bestDt = std::abs(timeOf(v[0]) - t);
    for (std::size_t i = 1; i < v.size(); ++i)
    {
        const double dt = std::abs(timeOf(v[i]) - t);
        if (dt < bestDt)
        {
            bestDt = dt;
            best = i;
        }
    }
    return best;
}

/// @brief Project a matrix onto SO(3) (nearest rotation in the Frobenius sense).
Eigen::Matrix3d orthonormalise(const Eigen::Matrix3d & M)
{
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
    D(2, 2) = (svd.matrixU()*svd.matrixV().transpose()).determinant() < 0.0 ? -1.0 : 1.0;
    return svd.matrixU()*D*svd.matrixV().transpose();
}

}   // namespace

std::size_t NeckKinematics::calibrate(const std::vector<SensorsSample> & sensors,
                                      const std::vector<VisionSample> & vision,
                                      double maxRate)
{
    Eigen::Matrix3d sum = Eigen::Matrix3d::Zero();
    std::vector<Eigen::Matrix3d> used;
    used.reserve(vision.size());

    for (const VisionSample & v : vision)
    {
        if (!v.Hcw.rotationMatrix.allFinite())
        {
            continue;
        }
        const std::size_t k = nearest(sensors, v.t, [](const SensorsSample & s) { return s.t; });
        if (k == std::size_t(-1))
        {
            continue;
        }
        const SensorsSample & s = sensors[k];
        // Only frames where the robot is barely turning: the logged extrinsic's
        // error is proportional to the body rate, so this is where it is closest
        // to the truth the servos describe directly.
        if (!s.headValid || !s.Htw.rotationMatrix.allFinite()
            || !s.gyroscope.allFinite() || s.gyroscope.norm() > maxRate)
        {
            continue;
        }
        const Eigen::Matrix3d RbcLog = s.Htw.rotationMatrix*v.Hcw.rotationMatrix.transpose();
        const Eigen::Matrix3d M = neckRotation(s.headYaw, s.headPitch).transpose()*RbcLog;
        used.push_back(M);
        sum += M;
    }

    if (used.empty())
    {
        calibrated_ = false;
        return 0;
    }

    // Chordal mean, then one robust re-average. The mean is pulled by frames
    // where the logged extrinsic is badly skewed -- a rate gate cannot catch all
    // of them, because the skew is between two message clocks rather than a
    // property of the motion -- and on `data`, whose head genuinely scans, the
    // raw mean lands 12 deg from the answer the bulk of the frames agree on.
    // Rejecting at three times the median deviation is enough: the good frames
    // agree to a few hundredths of a degree, so the two populations are not
    // close.
    mount_ = orthonormalise(sum/static_cast<double>(used.size()));

    std::vector<double> dev;
    dev.reserve(used.size());
    for (const Eigen::Matrix3d & M : used)
    {
        dev.push_back(logSO3(Eigen::Matrix3d(mount_.transpose()*orthonormalise(M))).norm());
    }
    std::vector<double> sorted = dev;
    std::nth_element(sorted.begin(), sorted.begin() + sorted.size()/2, sorted.end());
    const double median = sorted[sorted.size()/2];
    const double cutoff = std::max(3.0*median, 1e-3);   // never tighter than 0.06 deg

    Eigen::Matrix3d keptSum = Eigen::Matrix3d::Zero();
    std::size_t nKept = 0;
    for (std::size_t i = 0; i < used.size(); ++i)
    {
        if (dev[i] <= cutoff)
        {
            keptSum += used[i];
            nKept++;
        }
    }
    if (nKept >= 8)
    {
        mount_ = orthonormalise(keptSum/static_cast<double>(nKept));
    }
    calibrated_ = true;

    // Reported over the frames the mount was actually fitted to. The value does
    // not go to zero even for a perfect chain: the logged extrinsic it is
    // measured against carries the clock skew, and that is the whole reason this
    // class exists.
    double sumSq = 0.0;
    std::size_t nScored = 0;
    for (const Eigen::Matrix3d & M : used)
    {
        const double a = logSO3(Eigen::Matrix3d(mount_.transpose()*orthonormalise(M))).norm();
        if (a <= cutoff)
        {
            sumSq += a*a;
            nScored++;
        }
    }
    residualRms_ = nScored > 0 ? std::sqrt(sumSq/nScored) : 0.0;
    return nKept >= 8 ? nKept : used.size();
}
