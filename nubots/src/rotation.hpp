#ifndef ROTATION_HPP
#define ROTATION_HPP

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

#endif
