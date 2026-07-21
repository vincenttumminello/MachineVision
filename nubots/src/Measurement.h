/**
 * @file Measurement.h
 * @brief Defines the base Measurement class for handling measurement events in the system.
 */
#ifndef MEASUREMENT_H
#define MEASUREMENT_H

#include <limits>
#include <Eigen/Core>
#include "SystemBase.h"
#include "SystemEstimator.h"
#include "Event.h"

/**
 * @class Measurement
 * @brief Base class for measurement events in the system.
 *
 * This class represents a measurement event that can be processed by the system.
 * It inherits from the Event class and adds functionality specific to measurements.
 */
class Measurement : public Event
{
public:
    /**
     * @brief Construct a new Measurement object.
     * @param time The time at which the measurement occurs.
     */
    Measurement(double time);

    /**
     * @brief Construct a new Measurement object.
     * @param time The time at which the measurement occurs.
     * @param verbosity The verbosity level for logging and output.
     */
    Measurement(double time, int verbosity);

    /**
     * @brief Destroy the Measurement object.
     */
    virtual ~Measurement() override;

    /**
     * @brief Simulate a measurement given a state and system.
     * @param x The state vector.
     * @param system The system estimator.
     * @return The simulated measurement.
     */
    virtual Eigen::VectorXd simulate(const Eigen::VectorXd & x, const SystemEstimator & system) const = 0;

    /**
     * @brief Calculate the log-likelihood of a measurement.
     * @param x The state vector.
     * @param system The system estimator.
     * @return The log-likelihood value.
     */
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system) const = 0;

    /**
     * @brief Calculate the log-likelihood and its gradient.
     * @param x The state vector.
     * @param system The system estimator.
     * @param g Output parameter for the gradient.
     * @return The log-likelihood value.
     */
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g) const = 0;

    /**
     * @brief Calculate the log-likelihood, its gradient, and Hessian.
     * @param x The state vector.
     * @param system The system estimator.
     * @param g Output parameter for the gradient.
     * @param H Output parameter for the Hessian.
     * @return The log-likelihood value.
     */
    virtual double logLikelihood(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g, Eigen::MatrixXd & H) const = 0;

    /**
     * @brief Laplace approximation of the log evidence log p(y) from the last update.
     *
     * log p(y) ~= -V(x*) + (n/2) log(2 pi) - sum(log|diag(Xi)|)
     * where V is the negative log joint density at the MAP estimate x* and Xi is
     * the posterior sqrt information matrix. Useful for hypothesis weighting and
     * update health monitoring. NaN until update() has run.
     */
    double logEvidence() const { return logEvidence_; }

    /**
     * @brief Re-run data association against the system's current density.
     *
     * No-op by default. The multi-hypothesis path calls this once per mixture
     * component (with the density set to that component) so a pose-dependent
     * association is recomputed for each hypothesis rather than shared from the
     * representative. Measurements with no data association ignore it.
     */
    virtual void reassociate(const SystemEstimator &) {}

protected:
    /**
     * @brief Calculate the cost of the joint density.
     * @param x The state vector.
     * @param system The system estimator.
     * @return The cost value.
     */
    double costJointDensity(const Eigen::VectorXd & x, const SystemEstimator & system) const;

    /**
     * @brief Calculate the cost of the joint density and its gradient.
     * @param x The state vector.
     * @param system The system estimator.
     * @param g Output parameter for the gradient.
     * @return The cost value.
     */
    double costJointDensity(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g) const;

    /**
     * @brief Calculate the cost of the joint density, its gradient, and Hessian.
     * @param x The state vector.
     * @param system The system estimator.
     * @param g Output parameter for the gradient.
     * @param H Output parameter for the Hessian.
     * @return The cost value.
     */
    double costJointDensity(const Eigen::VectorXd & x, const SystemEstimator & system, Eigen::VectorXd & g, Eigen::MatrixXd & H) const;

    /**
     * @brief Update the system based on this measurement.
     * @param system The system to update.
     */
    virtual void update(SystemBase & system) override;

    /**
     * @brief Enumeration of update methods.
     */
    enum class UpdateMethod {BFGSTRUSTSQRT, BFGSLMSQRT, SR1TRUSTEIG, NEWTONTRUSTEIG, AFFINE, GAUSSNEWTON, LEVENBERGMARQUARDT};

    UpdateMethod updateMethod_;  ///< The method used for updating the system.

    double logEvidence_ = std::numeric_limits<double>::quiet_NaN();  ///< Laplace log evidence from the last update.
};

#endif
