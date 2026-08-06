/**
 * @file SystemEstimator.h
 * @brief Defines the SystemEstimator class for state estimation.
 */

#ifndef SYSTEMESTIMATOR_H
#define SYSTEMESTIMATOR_H

#include <cstddef>
#include <vector>
#include <Eigen/Core>
#include "GaussianInfo.hpp"
#include "SystemBase.h"

/**
 * @class SystemEstimator
 * @brief Class for system state estimation.
 *
 * This class extends SystemBase to provide state estimation functionality.
 */
class SystemEstimator : public SystemBase
{
public:
    /**
     * @brief Construct a new SystemEstimator object with initial density.
     * @param density Initial state density.
     */
    SystemEstimator(const GaussianInfo<double> & density);

    /**
     * @brief Destroy the SystemEstimator object.
     */
    virtual ~SystemEstimator() override;

    /**
     * @brief Predict the system state at a given time.
     * @param time The time to predict the system state for.
     */
    virtual void predict(double time) override;

    /**
     * @brief Number of predicts rejected for arriving out of sequence (dt < 0).
     *
     * Always zero for a correctly ordered event stream. Non-zero means something
     * upstream delivered a measurement stamped earlier than one already applied,
     * and that measurement was folded in at the wrong time.
     */
    std::size_t backwardPredicts() const { return nBackwardPredicts_; }

    /**
     * @brief Largest backward step seen, i.e. the worst out-of-sequence lag [s].
     */
    double maxBackwardDt() const { return maxBackwardDt_; }

    GaussianInfo<double> density;  ///< The current state density estimate.

    /**
     * @brief Compute the estimated system dynamics.
     * @param t Time.
     * @param x State vector.
     * @return The computed dynamics (state derivative).
     */
    virtual Eigen::VectorXd dynamicsEst(double t, const Eigen::VectorXd & x) const;

    /**
     * @brief Compute the estimated system dynamics and its Jacobian.
     * @param t Time.
     * @param x State vector.
     * @param J Output parameter for the Jacobian matrix.
     * @return The computed dynamics (state derivative).
     */
    virtual Eigen::VectorXd dynamicsEst(double t, const Eigen::VectorXd & x, Eigen::MatrixXd & J) const;

protected:
    /**
     * @brief Compute the augmented dynamics estimate.
     * @param t Time.
     * @param X The augmented state matrix.
     * @return Eigen::MatrixXd The augmented dynamics.
     */
    Eigen::MatrixXd augmentedDynamicsEst(double t, const Eigen::MatrixXd & X) const;

    /**
     * @brief Helper function for Runge-Kutta 4th order method for SDEs.
     * @param xdw The state and noise vector.
     * @param dt The time step.
     * @param J Output parameter for the Jacobian matrix.
     * @return The updated state vector.
     */
    Eigen::VectorXd RK4SDEHelper(const Eigen::VectorXd & xdw, double dt, Eigen::MatrixXd & J) const;

    /**
     * @brief Compute the process noise density.
     * @param dt The time step.
     * @return The process noise density.
     */
    virtual GaussianInfo<double> processNoiseDensity(double dt) const = 0;

    /**
     * @brief Get the indices of state variables affected by process noise.
     * @return The indices of affected state variables.
     */
    virtual std::vector<Eigen::Index> processNoiseIndex() const = 0;

    double dtMaxEst = 1e-2;         ///< Maximum time step for process model prediction

    std::size_t nBackwardPredicts_ = 0;   ///< Events rejected for arriving out of sequence
    double maxBackwardDt_ = 0.0;          ///< Worst out-of-sequence lag seen [s]
};

#endif
