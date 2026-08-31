#ifndef ROTATION_HPP
#define ROTATION_HPP

#include <algorithm>    // logSO3 clamps the trace before the arccosine
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>   // tangentBasis uses cross products

template <typename Scalar>
Eigen::Matrix3<Scalar> rotx(const Scalar & x)
{
    using std::cos, std::sin;
    Eigen::Matrix3<Scalar> R = Eigen::Matrix3<Scalar>::Identity();
    R(1, 1) = cos(x);
    R(1, 2) = -sin(x);
    R(2, 1) = sin(x);
    R(2, 2) = cos(x);
    return R;
}

template <typename Scalar>
Eigen::Matrix3<Scalar> rotx(const Scalar & x, Eigen::Matrix3<Scalar> & dRdx)
{
    using std::cos, std::sin;
    dRdx            =  Eigen::Matrix3<Scalar>::Zero();

    dRdx(1,1)       = -sin(x);
    dRdx(2,1)       =  cos(x);

    dRdx(1,2)       = -cos(x);
    dRdx(2,2)       = -sin(x);
    return rotx(x);
}

template <typename Scalar>
Eigen::Matrix3<Scalar> roty(const Scalar & x)
{
    using std::cos, std::sin;
    Eigen::Matrix3<Scalar> R = Eigen::Matrix3<Scalar>::Identity();
    R(0, 0) = cos(x);
    R(0, 2) = sin(x);
    R(2, 0) = -sin(x);
    R(2, 2) = cos(x);
    return R;
}

template <typename Scalar>
Eigen::Matrix3<Scalar> roty(const Scalar & x, Eigen::Matrix3<Scalar> & dRdx)
{
    using std::cos, std::sin;
    dRdx         =  Eigen::Matrix3<Scalar>::Zero();

    dRdx(0,0)    = -sin(x);
    dRdx(2,0)    = -cos(x);

    dRdx(0,2)    =  cos(x);
    dRdx(2,2)    = -sin(x);
    return roty(x);
}

template <typename Scalar>
Eigen::Matrix3<Scalar> rotz(const Scalar & x)
{
    using std::cos, std::sin;
    Eigen::Matrix3<Scalar> R = Eigen::Matrix3<Scalar>::Identity();
    R(0, 0) = cos(x);
    R(0, 1) = -sin(x);
    R(1, 0) = sin(x);
    R(1, 1) = cos(x);
    return R;
}

template <typename Scalar>
Eigen::Matrix3<Scalar> rotz(const Scalar & x, Eigen::Matrix3<Scalar> & dRdx)
{
    using std::cos, std::sin;
    dRdx         =  Eigen::Matrix3<Scalar>::Zero();

    dRdx(0,0)    = -sin(x);
    dRdx(1,0)    =  cos(x);

    dRdx(0,1)    = -cos(x);
    dRdx(1,1)    = -sin(x);
    return rotz(x);
}

template <typename Derived>
Eigen::Matrix3<typename Derived::Scalar> rpy2rot(const Eigen::MatrixBase<Derived> & Theta)
{
    using Scalar = typename Derived::Scalar;
    // R = Rz*Ry*Rx
    Eigen::Matrix3<Scalar> R;
    R = rotz(Theta(2)) * roty(Theta(1)) * rotx(Theta(0));
    return R;
}

template <typename Derived>
Eigen::Vector3<typename Derived::Scalar> rot2rpy(const Eigen::MatrixBase<Derived> & R)
{
    using Scalar = typename Derived::Scalar;
    using std::atan2, std::hypot;
    Eigen::Vector3<Scalar> Theta;
    Theta(0) = atan2(R(2, 1), R(2, 2));
    Theta(1) = atan2(-R(2, 0), hypot(R(2, 1), R(2, 2)));
    Theta(2) = atan2(R(1, 0), R(0, 0));
    return Theta;
}

/**
 * @brief Rotation matrix from a quaternion (w, x, y, z).
 *
 * The quaternion is normalised inside, which is what makes every geometric model
 * built on this function invariant to |q|. That invariance is deliberate: it
 * confines the redundant fourth degree of freedom to a direction no bearing,
 * gravity or height measurement can see, so it cannot corrupt the attitude
 * estimate. MeasurementQuaternionNorm is what supplies information along it, and
 * without that the MAP Hessian would be singular there.
 */
template <typename Derived>
Eigen::Matrix3<typename Derived::Scalar> quat2rot(const Eigen::MatrixBase<Derived> & q)
{
    using Scalar = typename Derived::Scalar;
    using std::sqrt;

    const Scalar n = sqrt(q(0)*q(0) + q(1)*q(1) + q(2)*q(2) + q(3)*q(3));
    const Scalar w = q(0)/n, x = q(1)/n, y = q(2)/n, z = q(3)/n;

    Eigen::Matrix3<Scalar> R;
    R(0, 0) = Scalar(1) - Scalar(2)*(y*y + z*z);
    R(0, 1) = Scalar(2)*(x*y - z*w);
    R(0, 2) = Scalar(2)*(x*z + y*w);
    R(1, 0) = Scalar(2)*(x*y + z*w);
    R(1, 1) = Scalar(1) - Scalar(2)*(x*x + z*z);
    R(1, 2) = Scalar(2)*(y*z - x*w);
    R(2, 0) = Scalar(2)*(x*z - y*w);
    R(2, 1) = Scalar(2)*(y*z + x*w);
    R(2, 2) = Scalar(1) - Scalar(2)*(x*x + y*y);
    return R;
}

/**
 * @brief Quaternion kinematics matrix: qdot = 0.5*quatXi(q)*omega_body.
 *
 * Follows from qdot = 0.5*q (x) (0, omega_b), the body-rate form matching
 * Rdot = R*hatSO3(omega_b) for R = quat2rot(q). Unlike the roll-pitch-yaw rate
 * transform it has no singularity: every entry is linear in q, so a robot
 * toppling through pitch = +-90 deg is an ordinary point on the trajectory.
 */
template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 4, 3> quatXi(const Eigen::MatrixBase<Derived> & q)
{
    using Scalar = typename Derived::Scalar;
    const Scalar & w = q(0);
    const Scalar & x = q(1);
    const Scalar & y = q(2);
    const Scalar & z = q(3);

    Eigen::Matrix<Scalar, 4, 3> Xi;
    Xi << -x, -y, -z,
           w, -z,  y,
           z,  w, -x,
          -y,  x,  w;
    return Xi;
}

/// @brief Hamilton product of two (w, x, y, z) quaternions.
template <typename DerivedA, typename DerivedB>
Eigen::Vector4<typename DerivedA::Scalar> quatMultiply(const Eigen::MatrixBase<DerivedA> & a,
                                                       const Eigen::MatrixBase<DerivedB> & b)
{
    using Scalar = typename DerivedA::Scalar;
    Eigen::Vector4<Scalar> c;
    c(0) = a(0)*b(0) - a(1)*b(1) - a(2)*b(2) - a(3)*b(3);
    c(1) = a(0)*b(1) + a(1)*b(0) + a(2)*b(3) - a(3)*b(2);
    c(2) = a(0)*b(2) - a(1)*b(3) + a(2)*b(0) + a(3)*b(1);
    c(3) = a(0)*b(3) + a(1)*b(2) - a(2)*b(1) + a(3)*b(0);
    return c;
}

/// @brief Quaternion (w, x, y, z) from a rotation matrix.
inline Eigen::Vector4d rot2quat(const Eigen::Matrix3d & R)
{
    const Eigen::Quaterniond q(R);
    // Eigen stores (x, y, z, w); the sign is free, so pick w >= 0 for a
    // canonical representative -- q and -q are the same rotation, and letting
    // the mean drift between the two hemispheres would make a Gaussian over the
    // components meaningless.
    Eigen::Vector4d v(q.w(), q.x(), q.y(), q.z());
    if (v(0) < 0.0) v = -v;
    return v.normalized();
}

/// @brief Quaternion (w, x, y, z) from roll-pitch-yaw.
inline Eigen::Vector4d rpy2quat(const Eigen::Vector3d & Theta)
{
    return rot2quat(rpy2rot(Theta));
}

template <typename Derived>
Eigen::Matrix3<typename Derived::Scalar> hatSO3(const Eigen::MatrixBase<Derived> & u)
{
    using Scalar = typename Derived::Scalar;
    const Scalar & u1 = u(0);
    const Scalar & u2 = u(1);
    const Scalar & u3 = u(2);
    Eigen::Matrix3<Scalar> S;
    S <<     0, -u3,  u2,
            u3,   0, -u1,
           -u2,  u1,   0;
    return S;
}

/**
 * @brief Orthonormal basis of the plane normal to a unit vector.
 *
 * The two columns span the tangent plane at u, which is where bearing
 * residuals and their covariances live: a unit ray has only two degrees of
 * freedom, so association surprisals are evaluated there rather than in the
 * rank-deficient 3D chordal space.
 *
 * @param u Unit vector
 * @return 3x2 matrix whose columns are orthonormal and perpendicular to u
 */
inline Eigen::Matrix<double, 3, 2> tangentBasis(const Eigen::Vector3d & u)
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

/**
 * @brief Exponential map from a rotation vector to SO(3) (Rodrigues' formula).
 *
 * expSO3(w) is the rotation of |w| radians about w/|w|, and is the inverse of
 * logSO3 below. Templated on the scalar so autodiff can differentiate through
 * it: the small-angle branch exists because sin(t)/t and (1 - cos t)/t^2 are
 * 0/0 at the origin, which is not merely inaccurate but a NaN in the value and
 * in every derivative that passes through it. The series are used well before
 * the closed forms lose precision, so the two agree to machine epsilon at the
 * switch.
 *
 * @param w Rotation vector (axis times angle) [rad]
 * @return The corresponding rotation matrix
 */
template <typename Derived>
Eigen::Matrix3<typename Derived::Scalar> expSO3(const Eigen::MatrixBase<Derived> & w)
{
    using Scalar = typename Derived::Scalar;
    using std::sqrt, std::sin, std::cos;

    const Scalar theta2 = w.squaredNorm();
    const Eigen::Matrix3<Scalar> W = hatSO3(w);

    Scalar a, b;    // sin(theta)/theta and (1 - cos(theta))/theta^2
    if (theta2 < Scalar(1e-8))
    {
        a = Scalar(1) - theta2/Scalar(6);
        b = Scalar(0.5) - theta2/Scalar(24);
    }
    else
    {
        const Scalar theta = sqrt(theta2);
        a = sin(theta)/theta;
        b = (Scalar(1) - cos(theta))/theta2;
    }
    return Eigen::Matrix3<Scalar>::Identity() + a*W + b*W*W;
}

/**
 * @brief Logarithm map from SO(3) to a rotation vector.
 *
 * The inverse of expSO3 for rotations of less than pi. Used to report an
 * inter-frame rotation as an angular velocity, and by the unit tests; the
 * measurement models themselves only ever need the forward map.
 *
 * @param R Rotation matrix
 * @return Rotation vector (axis times angle) [rad]
 */
inline Eigen::Vector3d logSO3(const Eigen::Matrix3d & R)
{
    // Antisymmetric part of R is sin(theta)*axis, trace gives cos(theta).
    const Eigen::Vector3d v(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));
    const double s = 0.5*v.norm();                                  // |sin(theta)|
    const double c = std::clamp(0.5*(R.trace() - 1.0), -1.0, 1.0);  // cos(theta)
    const double theta = std::atan2(s, c);

    if (s < 1e-8)
    {
        // Near identity (theta ~ 0): v/2 IS the rotation vector to first order.
        // Near a half turn (theta ~ pi) the antisymmetric part vanishes and the
        // axis has to come from the symmetric part instead. Neither case arises
        // for an inter-frame rotation at video rate, but returning a silently
        // wrong axis in the pi case would be worse than the cost of handling it.
        if (c > 0.0)
        {
            return 0.5*v;
        }
        const Eigen::Matrix3d A = 0.5*(R + Eigen::Matrix3d::Identity());   // = axis*axis'
        Eigen::Index i;
        A.diagonal().maxCoeff(&i);
        Eigen::Vector3d axis = A.col(i)/std::sqrt(std::max(A(i, i), 1e-12));
        return theta*axis.normalized();
    }
    return (0.5*theta/s)*v;
}

#endif
