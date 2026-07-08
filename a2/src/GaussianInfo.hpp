/**
 * @file GaussianInfo.hpp
 * @brief Defines the GaussianInfo class for representing Gaussian distributions using square-root information form.
 *
 * This file contains the implementation of the GaussianInfo class, which represents
 * a multivariate Gaussian (normal) distribution. The class provides various
 * operations such as marginalization, conditioning, and transformation of
 * Gaussian distributions. It also includes methods for computing log-likelihoods,
 * gradients, and Hessians, as well as utilities for working with confidence regions.
 *
 * The GaussianInfo class is templated on the scalar type, allowing for flexibility
 * in the numeric precision used for calculations and to enable autodiff support.
 */

#ifndef GAUSSIANINFO_HPP
#define GAUSSIANINFO_HPP

#include <cstddef>
#include <cmath>
#include <ctime>
#include <numbers>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/QR>
#include <Eigen/Cholesky>
#include <Eigen/LU> // TODO: Remove this header after fixing GaussianInfo::affineTransform to pass all unit tests
#include "GaussianBase.hpp"

/**
 * @brief Represents a Gaussian distribution using the information form.
 * 
 * @tparam Scalar The scalar type used for calculations (default: double).
 */
template <typename Scalar = double>
class GaussianInfo : public GaussianBase<Scalar>
{
public:
    virtual ~GaussianInfo() override = default;

    using GaussianBase<Scalar>::normcdf;
    using GaussianBase<Scalar>::chi2inv;
protected:
    /**
     * @brief Default constructor.
     */
    GaussianInfo()
        : GaussianBase<Scalar>()
    {}

    /**
     * @brief Constructor with dimension.
     * @param n The dimension of the Gaussian distribution.
     */
    explicit GaussianInfo(std::size_t n)
        : GaussianBase<Scalar>()
        , nu_(n)
        , Xi_(n, n)
    {}

    /**
     * @brief Constructor with square root information matrix.
     * @param Xi The square root information matrix.
     */
    explicit GaussianInfo(const Eigen::MatrixX<Scalar> & Xi)
        : GaussianBase<Scalar>()
        , nu_(Eigen::VectorX<Scalar>::Zero(Xi.cols()))
        , Xi_(Xi)
    {
        assert(nu_.size() == Xi_.cols());
        assert(Xi_.isUpperTriangular());
    }

    /**
     * @brief Constructor with square root information vector and matrix.
     * @param nu The square root information vector.
     * @param Xi The square root information matrix.
     */
    GaussianInfo(const Eigen::VectorX<Scalar> & nu, const Eigen::MatrixX<Scalar> & Xi)
        : GaussianBase<Scalar>()
        , nu_(nu)
        , Xi_(Xi)
    {
        assert(nu_.size() == Xi_.cols());
        assert(Xi_.isUpperTriangular());
    }

    /**
     * @brief Friend declaration to allow access to protected members for type conversion.
     */
    template <typename OtherScalar> friend class GaussianInfo;

    /**
     * @brief Copy constructor from a different scalar type.
     * @tparam OtherScalar The scalar type of the source GaussianInfo.
     * @param p The source GaussianInfo to copy from.
     */
    template <typename OtherScalar>
    explicit GaussianInfo(const GaussianInfo<OtherScalar> & p)
        : GaussianBase<Scalar>()
        , nu_(p.nu_.template cast<Scalar>())
        , Xi_(p.Xi_.template cast<Scalar>())
    {
        assert(nu_.size() == Xi_.cols());
        assert(Xi_.isUpperTriangular());
    }

public:
    /**
     * @brief Casts the GaussianInfo to a different scalar type.
     * 
     * @tparam OtherScalar The target scalar type.
     * @return The casted GaussianInfo<OtherScalar> object.
     */
    template <typename OtherScalar>
    GaussianInfo<OtherScalar> cast() const
    {
        return GaussianInfo<OtherScalar>(*this);
    }

    //
    // Two-argument factories
    //

    /**
     * @brief Creates a GaussianInfo object from square root moment parameters.
     * 
     * @param mu The mean vector.
     * @param S The square root of the covariance matrix (upper triangular).
     * @return The resulting GaussianInfo object.
     */
    static GaussianInfo fromSqrtMoment(const Eigen::VectorX<Scalar> & mu, const Eigen::MatrixX<Scalar> & S)
    {
        assert(mu.size() == S.cols());
        assert(S.isUpperTriangular());

        GaussianInfo out(S.cols());

        // qr(S^{-T})
        out.Xi_ = S.template triangularView<Eigen::Upper>().transpose().solve(
            Eigen::MatrixX<Scalar>::Identity(S.cols(), S.cols())
        );
        Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixX<Scalar>>> qr(out.Xi_);  // In-place QR decomposition
        out.Xi_ = out.Xi_.template triangularView<Eigen::Upper>();             // Safe aliasing

        // Xi*mu = nu
        out.nu_ = out.Xi_*mu;

        return out;
    }

    /**
     * @brief Creates a GaussianInfo object from moment parameters.
     * 
     * @param mu The mean vector.
     * @param P The covariance matrix.
     * @return The resulting GaussianInfo object.
     */
    static GaussianInfo fromMoment(const Eigen::VectorX<Scalar> & mu, const Eigen::MatrixX<Scalar> & P)
    {
        assert(mu.size() == P.cols());
        assert(P.rows() == P.cols());
        
        // Let S be an upper-triangular matrix such that S^T*S = P
        Eigen::LLT<Eigen::MatrixX<Scalar>, Eigen::Upper> llt(P);
        Eigen::MatrixX<Scalar> S = llt.matrixU();

        return fromSqrtMoment(mu, S);
    }

    /**
     * @brief Creates a GaussianInfo object from square root information parameters.
     * 
     * @param nu The square root information vector.
     * @param Xi The square root information matrix (upper triangular).
     * @return The resulting GaussianInfo object.
     */
    static GaussianInfo fromSqrtInfo(const Eigen::VectorX<Scalar> & nu, const Eigen::MatrixX<Scalar> & Xi)
    {
        assert(nu.size() == Xi.cols());
        assert(Xi.isUpperTriangular());

        GaussianInfo out(Xi.cols());
        out.nu_ = nu;
        out.Xi_ = Xi;
        return out;
    }

    /**
     * @brief Creates a GaussianInfo object from information parameters.
     * 
     * @param eta The information vector.
     * @param Lambda The information matrix.
     * @return The resulting GaussianInfo object.
     */
    static GaussianInfo fromInfo(const Eigen::VectorX<Scalar> & eta, const Eigen::MatrixX<Scalar> & Lambda)
    {
        assert(eta.size() == Lambda.cols());
        assert(Lambda.rows() == Lambda.cols());

        // Let Xi be an upper-triangular matrix such that Xi^T*Xi = Lambda
        Eigen::LLT<Eigen::MatrixX<Scalar>, Eigen::Upper> llt(Lambda);
        Eigen::MatrixX<Scalar> Xi = llt.matrixU();

        // Solve Xi^T*nu = eta
        Eigen::VectorX<Scalar> nu = Xi.template triangularView<Eigen::Upper>().transpose().solve(eta);
        return fromSqrtInfo(nu, Xi);
    }
    
    //
    // One-argument factories
    //

    /**
     * @brief Creates a GaussianInfo object from the square root of the covariance matrix.
     * 
     * This static factory method creates a GaussianInfo object with zero mean and
     * the given square root of the covariance matrix.
     * 
     * @param S The square root of the covariance matrix (upper triangular).
     * @return The resulting GaussianInfo object.
     */
    static GaussianInfo fromSqrtMoment(const Eigen::MatrixX<Scalar> & S)
    {
        return fromSqrtMoment(Eigen::VectorX<Scalar>::Zero(S.cols()), S);
    }

    /**
     * @brief Creates a GaussianInfo object from the covariance matrix.
     * 
     * This static factory method creates a GaussianInfo object with zero mean and
     * the given covariance matrix.
     * 
     * @param P The covariance matrix.
     * @return The resulting GaussianInfo object.
     */
    static GaussianInfo fromMoment(const Eigen::MatrixX<Scalar> & P)
    {
        return fromMoment(Eigen::VectorX<Scalar>::Zero(P.cols()), P);
    }

    /**
     * @brief Creates a GaussianInfo object from the square root information matrix.
     * 
     * This static factory method creates a GaussianInfo object with zero mean and
     * the given square root information matrix.
     * 
     * @param Xi The square root information matrix (upper triangular).
     * @return The resulting GaussianInfo object.
     */
    static GaussianInfo fromSqrtInfo(const Eigen::MatrixX<Scalar> & Xi)
    {
        return fromSqrtInfo(Eigen::VectorX<Scalar>::Zero(Xi.cols()), Xi);
    }

    /**
     * @brief Creates a GaussianInfo object from the information matrix.
     * 
     * This static factory method creates a GaussianInfo object with zero mean and
     * the given information matrix.
     * 
     * @param Lambda The information matrix.
     * @return The resulting GaussianInfo object.
     */
    static GaussianInfo fromInfo(const Eigen::MatrixX<Scalar> & Lambda)
    {
        return fromInfo(Eigen::VectorX<Scalar>::Zero(Lambda.cols()), Lambda);
    }

    /**
     * @brief Get the dimension of the Gaussian distribution.
     * 
     * @return The dimension of the distribution.
     */
    virtual Eigen::Index dim() const override
    {
        return Xi_.cols();
    }

    /**
     * @brief Get the mean of the Gaussian distribution.
     * 
     * @return The mean vector of the distribution.
     */
    virtual Eigen::VectorX<Scalar> mean() const override
    {
        // Solve Xi*mu = nu for mu
        return Xi_.template triangularView<Eigen::Upper>().solve(nu_);
    }

    /**
     * @brief Get the square root of the covariance matrix.
     * 
     * This method computes and returns the upper triangular square root of the covariance matrix,
     * also known as the Cholesky factor.
     * 
     * @return The upper triangular square root of the covariance matrix.
     */
    virtual Eigen::MatrixX<Scalar> sqrtCov() const override
    {
        // S = qr(Xi^{-T})
        Eigen::MatrixX<Scalar> S = Xi_.template triangularView<Eigen::Upper>().transpose().solve(
            Eigen::MatrixX<Scalar>::Identity(Xi_.cols(), Xi_.cols())
        );
        Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixX<Scalar>>> qr(S);         // In-place QR decomposition
        S = S.template triangularView<Eigen::Upper>();                          // Safe aliasing
        return S;
    }

    /**
     * @brief Get the covariance matrix of the Gaussian distribution.
     * 
     * This method computes and returns the covariance matrix by multiplying
     * the square root of the covariance matrix with its transpose.
     * 
     * @return The covariance matrix of the distribution.
     */
    virtual Eigen::MatrixX<Scalar> cov() const override
    {
        const Eigen::MatrixX<Scalar> & S = sqrtCov();
        return S.transpose()*S;
    }

    /**
     * @brief Get the information matrix of the Gaussian distribution.
     * 
     * This method computes and returns the information matrix, which is
     * the inverse of the covariance matrix.
     * 
     * @return The information matrix of the distribution.
     */
    virtual Eigen::MatrixX<Scalar> infoMat() const override
    {
        return Xi_.transpose()*Xi_;
    }

    /**
     * @brief Get the information vector of the Gaussian distribution.
     * 
     * This method computes and returns the information vector, which is
     * related to the mean of the distribution.
     * 
     * @return The information vector of the distribution.
     */
    virtual Eigen::VectorX<Scalar> infoVec() const override
    {
        return Xi_.transpose()*nu_;
    }

    /**
     * @brief Get the square root of the information matrix.
     * 
     * This method returns the square root of the information matrix,
     * which is stored internally as Xi_.
     * 
     * @return The square root of the information matrix.
     */
    virtual Eigen::MatrixX<Scalar> sqrtInfoMat() const override
    {
        return Xi_;
    }

    /**
     * @brief Get the square root of the information vector.
     * 
     * This method returns the square root of the information vector,
     * which is stored internally as nu_.
     * 
     * @return The square root of the information vector.
     */
    virtual Eigen::VectorX<Scalar> sqrtInfoVec() const override
    {
        return nu_;
    }

    /**
     * @brief Create a GaussianInfo object from a set of samples.
     * 
     * This static method constructs a GaussianInfo object by estimating the
     * mean and square-root covariance from the provided sample data.
     * 
     * @param X The sample data matrix, where each column represents a sample
     *          and each row represents a dimension of the data.
     * @return A GaussianInfo object representing the estimated Gaussian distribution.
     * 
     * @note The method assumes that the samples are stored column-wise in the input matrix.
     */
    static GaussianInfo fromSamples(const Eigen::MatrixX<Scalar> & X)
    {
        const Eigen::Index n = X.rows();
        const Eigen::Index m = X.cols();

        // Compute the sample mean
        Eigen::VectorXd mu = X.rowwise().mean();

        // Compute the sample square-root covariance
        Eigen::MatrixX<Scalar> SS = std::sqrt(1.0 / (m - 1))*(X.colwise() - mu).transpose();
        Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixX<Scalar>>> qr(SS);   // In-place QR decomposition
        Eigen::MatrixXd S = SS.topRows(n).template triangularView<Eigen::Upper>();
        return GaussianInfo::fromSqrtMoment(mu, S);
    }

    /**
     * @brief Given joint density p(x), return marginal density p(x(idx))
     * 
     * This method computes the marginal density for a subset of variables specified by idx.
     * 
     * @tparam IndexType The type of the index container
     * @tparam NotIndexType The type of the complementary index container
     * @param idx The indices of the variables to keep in the marginal
     * @param idxNot The indices of the variables to marginalize out
     * @return The marginal Gaussian distribution
     */
    template <typename IndexType, typename NotIndexType>
    GaussianInfo marginal(const IndexType & idx, const NotIndexType & idxNot) const
    {
        const std::size_t & nI = idx.size();
        const std::size_t & nNotI = idxNot.size();
        const std::size_t n = nI + nNotI;
        assert(n == dim());

        // Form [Xi(:, idxNot), Xi(:, idx), nu]

        // Q-less QR yields
        // [R1, R2, nu1;
        //   0, R3, nu2]

        // p(x(idx)) = N^-0.5(x(idx); nu2, R3)
        GaussianInfo out(nI);
        // Form [Xi(:, idxNot), Xi(:, idx), nu]
        // This creates the matrix from equation (19): [Ξ_:,Ī, Ξ_:,I, ν]
        Eigen::MatrixX<Scalar> A(n, n + 1);
        
        // Fill in Xi(:, idxNot) - the columns we want to marginalize out
        for (std::size_t i = 0; i < nNotI; ++i) {
            A.col(i) = Xi_.col(idxNot[i]);
        }
        
        // Fill in Xi(:, idx) - the columns we want to keep
        for (std::size_t i = 0; i < nI; ++i) {
            A.col(nNotI + i) = Xi_.col(idx[i]);
        }
        
        // Fill in nu as the last column
        A.col(n) = nu_;

        // Q-less QR decomposition yields
        // [R1, R2, nu1;
        //   0, R3, nu2]
        // This corresponds to equation (19) where Q[R1 R2 nu1; 0 R3 nu2] = [Ξ_:,Ī, Ξ_:,I, ν]
        Eigen::HouseholderQR<Eigen::MatrixX<Scalar>> qr(A);
        Eigen::MatrixX<Scalar> R = qr.matrixQR().template triangularView<Eigen::Upper>();
        
        // Extract R3 and nu2 from the bottom-right partition
        // R3 is the bottom-right nI x nI block
        // nu2 is the bottom nI elements of the last column
        Eigen::MatrixX<Scalar> R3 = R.block(nNotI, nNotI, nI, nI);
        Eigen::VectorX<Scalar> nu2 = R.block(nNotI, n, nI, 1);

        // p(x(idx)) = N^-0.5(x(idx); nu2, R3)
        // From equation (21): p(x_I) = N^-1/2(x_I; r2, R3)
        out.nu_ = nu2;
        out.Xi_ = R3;

        return out;
    }

    /**
     * @brief Compute the marginal density for a subset of variables.
     * 
     * This method computes the marginal density for a subset of variables specified by idx.
     * It automatically computes the complementary indices.
     * 
     * @tparam IndexType The type of the index container
     * @param idx The indices of the variables to keep in the marginal
     * @return The marginal Gaussian distribution
     */
    template <typename IndexType>
    GaussianInfo marginal(const IndexType & idx) const
    {
        const std::size_t & n = dim();
        std::vector<bool> isNotInIdx(n, true);
        for (Eigen::Index ii = 0; ii < idx.size(); ++ii)
        {
            std::size_t i = idx[ii];
            isNotInIdx[i] = false;
        }

        // Complementary indices
        std::vector<int> idxNot;
        idxNot.reserve(n);                          // Reserve maximum possible size to avoid reallocation
        for (std::size_t i = 0; i < n; ++i)
        {
            if (isNotInIdx[i])
            {
                idxNot.push_back(i);
            }
        }

        return marginal(idx, idxNot);
    }

    /**
     * @brief Given joint density p(x), return conditional density p(x(idxA) | x(idxB) = xB)
     * 
     * This method computes the conditional density for a subset of variables given values for another subset.
     * 
     * @tparam IndexTypeA The type of the index container for variables A
     * @tparam IndexTypeB The type of the index container for variables B
     * @param idxA The indices of the variables to condition on
     * @param idxB The indices of the variables with known values
     * @param xB The known values for variables indexed by idxB
     * @return The conditional Gaussian distribution
     */
    template <typename IndexTypeA, typename IndexTypeB>
    GaussianInfo conditional(const IndexTypeA & idxA, const IndexTypeB & idxB, const Eigen::VectorX<Scalar> & xB) const
    {
        const std::size_t & nA = idxA.size();
        const std::size_t & nB = idxB.size();
        const std::size_t n = nA + nB;
        assert(n == dim());

        // Form [Xi(:, idxA), Xi(:, idxB), nu]

        // Q-less QR yields
        // [R1, R2, nu1;
        //   0, R3, nu2]

        // p(x(idxA) | x(idxB) = xB) = N^-0.5(x(idxA); nu1 - R2*xB, R1)
        // Form [Xi(:, idxA), Xi(:, idxB), nu]
        // This creates the matrix from equation (28): [Ξ_:,A, Ξ_:,B, ν]
        Eigen::MatrixX<Scalar> A(n, n + 1);
        
        // Fill in Xi(:, idxA) - the columns for variables we want to condition
        for (std::size_t i = 0; i < nA; ++i) {
            A.col(i) = Xi_.col(idxA[i]);
        }
        
        // Fill in Xi(:, idxB) - the columns for variables with known values
        for (std::size_t i = 0; i < nB; ++i) {
            A.col(nA + i) = Xi_.col(idxB[i]);
        }
        
        // Fill in nu as the last column
        A.col(n) = nu_;

        // Q-less QR decomposition yields
        // [R1, R2, nu1;
        //   0, R3, nu2]
        // This corresponds to equation (28) where Q[R1 R2 nu1; 0 R3 nu2] = [Ξ_:,A, Ξ_:,B, ν]
        Eigen::HouseholderQR<Eigen::MatrixX<Scalar>> qr(A);
        Eigen::MatrixX<Scalar> R = qr.matrixQR().template triangularView<Eigen::Upper>();
        
        // Extract R1, R2, and nu1 from the top partition
        // R1 is the top-left nA x nA block
        // R2 is the top-right nA x nB block  
        // nu1 is the top nA elements of the last column
        Eigen::MatrixX<Scalar> R1 = R.block(0, 0, nA, nA);
        Eigen::MatrixX<Scalar> R2 = R.block(0, nA, nA, nB);
        Eigen::VectorX<Scalar> nu1 = R.block(0, n, nA, 1);

        // p(x(idxA) | x(idxB) = xB) = N^-0.5(x(idxA); nu1 - R2*xB, R1)
        // From equation (30): p(x_A | x_B) = N^-1/2(x_A; r1 - R2*x_B, R1)
        GaussianInfo out(nA);
        out.nu_ = nu1 - R2 * xB;
        out.Xi_ = R1;
        return out;
    }

    /**
     * @brief Given joint density p(x), return conditional density p(x(idxA) | y) given p(x(idxB) | y) for some data y
     *
     * This method computes the conditional density for a subset of variables given the conditional density of another subset.
     *
     * @tparam IndexTypeA The type of the index container for variables A
     * @tparam IndexTypeB The type of the index container for variables B
     * @param idxA The indices of the variables to condition on
     * @param idxB The indices of the variables with known conditional density
     * @param pxB_y The conditional density p(x(idxB) | y)
     * @return The conditional Gaussian distribution p(x(idxA) | y)
     */
    template <typename IndexTypeA, typename IndexTypeB>
    GaussianInfo conditional(const IndexTypeA & idxA, const IndexTypeB & idxB, const GaussianInfo & pxB_y) const
    {
        const std::size_t & nA = idxA.size();
        const std::size_t & nB = idxB.size();
        const std::size_t n = nA + nB;
        assert(n == dim());

        // Form [Xi(:, idxA), Xi(:, idxB), nu]
    Eigen::MatrixX<Scalar> RR(n, n + 1);
        RR << Xi_(Eigen::all, idxA), Xi_(Eigen::all, idxB), nu_;
        // Q-less QR yields
        // [R1, R2, nu1;
        //   0, R3, nu2]
        Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixX<Scalar>>> qr(RR);        // In-place QR decomposition

        // Form [      R2,            R1,      nu1; 
        //       pxB_y.Xi, zeros(nB, nA), pxB_y.nu]
        Eigen::MatrixX<Scalar> SS(n, n + 1);
        SS.topLeftCorner(nA, nB)        = RR.block(0, nA, nA, nB); // R2
        SS.block(0, nB, nA, nA)         = RR.topLeftCorner(nA, nA).template triangularView<Eigen::Upper>(); // R1
        SS.topRightCorner(nA, 1)        = RR.block(0, n, nA, 1); // nu1
        SS.bottomLeftCorner(nB, nB)     = pxB_y.Xi_;
        SS.block(nA, nB, nB, nA)        = Eigen::MatrixX<Scalar>::Zero(nB, nA);
        SS.bottomRightCorner(nB, 1)     = pxB_y.nu_;
        // Q-less QR yields
        // [S1, S2, s1;
        //   0, S3, s2]
        Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixX<Scalar>>> qr_SS(SS);    // In-place QR decomposition

        // p(x(idxA) | y) = N^-0.5(x(idxA); s2, S3)
        GaussianInfo out(nA);
        out.nu_ = SS.block(nB, n, nA, 1);
        out.Xi_ = SS.block(nB, nB, nA, nA).template triangularView<Eigen::Upper>();
        return out;
    }

    /**
     * @brief Propagate the Gaussian distribution through a nonlinear function.
     *
     * This method transforms the current Gaussian distribution p(x) through a given function y = h(x)
     * by propagating information through the affine transformation. It returns a new Gaussian distribution
     * representing p(y).
     *
     * @tparam Func The type of the function object.
     * @param h The function object representing the nonlinear transformation.
     *          It should take two arguments: the input vector and a reference to the Jacobian matrix.
     *          The function should return the transformed vector and populate the Jacobian matrix.
     * @return A new Gaussian distribution representing p(y).
     */
    template <typename Func>
    GaussianInfo affineTransform(Func h) const
    {
        // Evaluate the function at the mean value and compute the Jacobian
        Eigen::MatrixX<Scalar> J;
        Eigen::VectorX<Scalar> mux = mean();
        Eigen::VectorX<Scalar> muy = h(mux, J);  // Evaluate function at mean value
        const std::size_t m = J.rows();
        const std::size_t n = J.cols();
        assert(m == muy.size());
        assert(n == dim());

        // Compute the offset vector b
        Eigen::VectorX<Scalar> b = muy - J * mux;

        // Perform SVD on the Jacobian matrix
        Eigen::JacobiSVD<Eigen::MatrixX<Scalar>> svd(J, Eigen::ComputeFullU | Eigen::ComputeFullV);
        const Eigen::MatrixX<Scalar>& U = svd.matrixU();
        const Eigen::MatrixX<Scalar>& V = svd.matrixV();
        const Eigen::VectorX<Scalar>& singularValues = svd.singularValues();

        // Determine the numerical rank
        const Scalar tolerance = std::numeric_limits<Scalar>::epsilon() * std::max(m, n) * singularValues(0);
        Eigen::Index r = 0;

        for (Eigen::Index i = 0; i < singularValues.size(); ++i) {
            if (singularValues(i) > tolerance) {
                r++;
            } else {
                break;
            }
        }

        // Extract rank-r components
        Eigen::MatrixX<Scalar> U1 = U.leftCols(r);
        Eigen::MatrixX<Scalar> U2 = U.rightCols(m - r);
        Eigen::MatrixX<Scalar> V1 = V.leftCols(r);
        Eigen::MatrixX<Scalar> V2 = V.rightCols(n - r);
        Eigen::VectorX<Scalar> sigma11 = singularValues.head(r);

        // Create sigma11
        Eigen::MatrixX<Scalar> Sigma11 = sigma11.asDiagonal();
        Eigen::MatrixX<Scalar> Sigma11_inv = Sigma11.inverse();

        // Use large penalty parameter kappa (similar to matlab)
        Eigen::MatrixX<Scalar> X = Xi_ * V2;
        Eigen::MatrixX<Scalar> Y = Xi_ * V1 * Sigma11_inv * U1.transpose();

        Eigen::MatrixX<Scalar> XY(X.rows(), X.cols() + Y.cols());
        XY << X, Y;
        Scalar sigma_max_ub = std::sqrt(XY.array().square().sum());
        const Scalar kappa = 1e6 * sigma_max_ub;
        

        // Create the augmented matrix for QR decomposition
        Eigen::MatrixX<Scalar> augmented(n + m - r, n + m + 1);  // Adjusted dimensions for rank-r components
        augmented.setZero();

        // Precompute
        Eigen::MatrixX<Scalar> XiV1SigmaInvU1T = Xi_ * V1 * Sigma11_inv * U1.transpose();

        // Fill top part: [ΞV1Σ⁻¹U1ᵀ, ν + ΞV1Σ⁻¹U1ᵀb]
        augmented.topLeftCorner(n, n - r) = Xi_ * V2;
        augmented.block(0, n - r, n, m) = XiV1SigmaInvU1T;
        augmented.block(0, n + r, n, 1) = nu_ + XiV1SigmaInvU1T * b;

        // Fill bottom part: [κU2ᵀ, κU2ᵀb]
        if (m > r) {
            augmented.block(n, n-r, m-r, m) = kappa * U2.transpose(); 
            augmented.block(n, n + r, m - r, 1) = kappa * U2.transpose() * b;
        }

        // Perform Q-less QR decomposition
        Eigen::HouseholderQR<Eigen::Ref<Eigen::MatrixX<Scalar>>> qr(augmented); // In-place QR decomposition

        // Extract the marginal for y (tail partition)
        // p(y) = N^(-1/2)(y; ν₂, R₃) from equation (42)
        GaussianInfo out(m);
        out.Xi_ = augmented.block(n-r, n-r, m, m).template triangularView<Eigen::Upper>();
        out.nu_ = augmented.block(n-r, n + r, m, 1);

        return out;
    }

    /**
     * @brief Compute the log-likelihood of a given vector.
     * 
     * This method calculates the log-likelihood of the vector x under the current
     * Gaussian distribution represented in information form.
     * 
     * @param x The input vector for which to compute the log-likelihood.
     * @return The log-likelihood value.
     */
    virtual Scalar log(const Eigen::VectorX<Scalar> & x) const override
    {
        assert(x.cols() == 1);
        assert(x.size() == dim());

        static const Scalar halflog2pi = std::log(2 * std::numbers::pi) / 2.0;
        const int n = dim();

        // Compute Z = Xi_ * x - nu_
        Eigen::VectorX<Scalar> Z = Xi_ * x - nu_;

        // Compute the log-determinant of Xi_ using the diagonal elements
        Scalar logdet = Xi_.diagonal().array().abs().log().sum();

        // Compute the quadratic term
        Scalar quad = Z.squaredNorm();

        // Combine terms to compute the log-likelihood
        return -halflog2pi * n + logdet - 0.5 * quad;
    }

    /**
     * @brief Compute the log-likelihood of a given vector and its gradient.
     * 
     * This method calculates the log-likelihood of the vector x under the current
     * Gaussian distribution represented in information form. It also computes
     * the gradient of the log-likelihood with respect to x.
     * 
     * @param x The input vector for which to compute the log-likelihood.
     * @param g Reference to a vector where the gradient will be stored.
     * @return The log-likelihood value.
     */
    Scalar log(const Eigen::VectorX<Scalar> & x, Eigen::VectorX<Scalar> & g) const
    {
        // Compute gradient g
        // Compute Z = Xi_ * x - nu_
        Eigen::VectorX<Scalar> Z = Xi_ * x - nu_;

        // Compute gradient g = -Xi_.transpose() * Z
        // From MATLAB: g = -obj.Xi.' * Z
        g = -Xi_.transpose() * Z;

        return log(x);
    }

    /**
     * @brief Compute the log-likelihood of a given vector, its gradient, and its Hessian.
     * 
     * This method calculates the log-likelihood of the vector x under the current
     * Gaussian distribution represented in information form. It also computes
     * the gradient and Hessian of the log-likelihood with respect to x.
     * 
     * @param x The input vector for which to compute the log-likelihood.
     * @param g Reference to a vector where the gradient will be stored.
     * @param H Reference to a matrix where the Hessian will be stored.
     * @return The log-likelihood value.
     */
    Scalar log(const Eigen::VectorX<Scalar> & x, Eigen::VectorX<Scalar> & g, Eigen::MatrixX<Scalar> & H) const
    {
        // Compute Hessian H
        // Compute Hessian H = -Xi_.transpose() * Xi_
        // From MATLAB: H = -obj.infoMat() where infoMat() = Xi.'*Xi
        // So H = -Xi_.transpose() * Xi_
        H = -Xi_.transpose() * Xi_;

        return log(x, g);
    }

    /**
     * @brief Join two Gaussian distributions into a joint distribution.
     *
     * This method combines the current Gaussian distribution with another one,
     * creating a joint distribution. The resulting distribution represents
     * the joint probability of both input distributions, assuming they are independent.
     *
     * @param other The other GaussianInfo object to join with.
     * @return A new GaussianInfo object representing the joint distribution.
     */
    GaussianInfo join(const GaussianInfo & other) const
    {
        const Eigen::Index & n1 = dim();
        const Eigen::Index & n2 = other.dim();
        GaussianInfo out(n1 + n2);
        out.nu_ << nu_, other.nu_;
        out.Xi_ << Xi_,                                  Eigen::MatrixX<Scalar>::Zero(n1, n2),
                   Eigen::MatrixX<Scalar>::Zero(n2, n1), other.Xi_;
        return out;
    }

    /**
     * @brief Multiply two Gaussian distributions to create a joint distribution.
     *
     * This operator overload combines the current Gaussian distribution with another one,
     * creating a joint distribution. The resulting distribution represents
     * the joint probability of both input distributions, assuming they are independent.
     *
     * @param other The other GaussianInfo object to multiply with.
     * @return A new GaussianInfo object representing the joint distribution.
     */
    GaussianInfo operator*(const GaussianInfo & other) const
    {
        return join(other);
    }

    /**
     * @brief Multiply and assign another Gaussian distribution to the current one.
     *
     * This operator multiplies the current Gaussian distribution with another one,
     * creating a joint distribution. The resulting distribution represents
     * the joint probability of both input distributions, assuming they are independent.
     * The result is stored in the current object.
     *
     * @param other The other GaussianInfo object to multiply with.
     * @return A reference to the current object after multiplication.
     */
    GaussianInfo & operator*=(const GaussianInfo & other)
    {
        const Eigen::Index & n1 = dim();
        const Eigen::Index & n2 = other.dim();
        nu_.conservativeResize(n1 + n2);
        nu_.tail(n2) = other.nu_;
        Xi_.conservativeResizeLike(Eigen::MatrixX<Scalar>::Zero(n1 + n2, n1 + n2));
        Xi_.bottomRightCorner(n2, n2) = other.Xi_;
        return *this;
    }

    /**
     * @brief Check if a given point is within the confidence region of the Gaussian distribution.
     *
     * This method determines whether the input vector x is within the confidence region
     * defined by nSigma standard deviations from the mean of the Gaussian distribution.
     *
     * @param x The input vector to check.
     * @param nSigma The number of standard deviations defining the confidence region (default: 3.0).
     * @return True if the point is within the confidence region, false otherwise.
     */
    virtual bool isWithinConfidenceRegion(const Eigen::VectorX<Scalar> & x, double nSigma = 3.0) const override
    {
        const Eigen::Index & n = dim();
        double c = 2.0 * normcdf(nSigma) - 1.0; // Two-tailed confidence level 
        Scalar chi2_threshold = static_cast<Scalar>(chi2inv(c, static_cast<int>(n)));

        Eigen::VectorX<Scalar> w = Xi_ * x - nu_;

        // Check if w'w <= chi2_threshold
        return w.squaredNorm() <= chi2_threshold;
    }

    /**
     * @brief Compute the quadric surface coefficients for a given number of standard deviations.
     *
     * This method calculates the coefficients of the quadric surface that represents
     * the confidence ellipsoid of the 3D Gaussian distribution. The surface is defined
     * for a specified number of standard deviations (nSigma).
     *
     * @param nSigma The number of standard deviations to use for the confidence ellipsoid (default: 3.0).
     * @return A 4x4 matrix containing the quadric surface coefficients.
     *
     * @note This method assumes that the Gaussian distribution is three-dimensional.
     */
    Eigen::Matrix4<Scalar> quadricSurface(double nSigma = 3.0) const
    {
        const Eigen::Index & n = dim();
        assert(n == 3);
        
        Eigen::Matrix4<Scalar> Q;
        // Probability mass enclosed by nSigma standard deviations
        // c = 2*normcdf(nSigma) - 1 is equivalent to normcdf(nSigma) - normcdf(-nSigma)
        double c = 2.0 * normcdf(nSigma) - 1.0;
        
        // Squared radius in w coordinates
        Scalar r2 = static_cast<Scalar>(chi2inv(c, static_cast<int>(n)));
        
        // Get the information vector and negate it: y = -obj.infoVec()
        Eigen::Vector3<Scalar> y = -infoVec();
        
        // Top-left 3x3: information matrix
        Q.topLeftCorner(3, 3) = infoMat();
        
        // Top-right 3x1: y vector
        Q.topRightCorner(3, 1) = y;
        
        // Bottom-left 1x3: y transpose
        Q.bottomLeftCorner(1, 3) = y.transpose();
        
        // Bottom-right 1x1: nu.'*nu - r2
        Q.bottomRightCorner(1, 1) << nu_.transpose() * nu_ - r2;

        return Q;
    }

protected:
    Eigen::VectorX<Scalar> nu_; ///< The square root information vector.
    Eigen::MatrixX<Scalar> Xi_; ///< The square root information matrix.
};

#endif

