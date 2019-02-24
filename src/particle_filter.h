/**
 * particle_filter.h
 * 2D particle filter class.
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#ifndef PARTICLE_FILTER_H_
#define PARTICLE_FILTER_H_

#include <string>
#include <vector>
#include "helper_functions.h"
#include "types.h"

class Particle : public CartesianPoint {
 public:
  Particle(double x, double y, double theta);
  const std::vector<LandmarkObs>& expected() const {
    return expected_;
  }
  double theta() const { return theta_; }
  double weight() const { return weight_; }
  void update_estimate(double x, double y, double theta);
  void update_weight(double weight);
  void update_expected(
      std::vector<LandmarkObs>&& expected);
  std::vector<int> associations();
  std::vector<double> sense_x();
  std::vector<double> sense_y();

 private:
  double theta_;
  double weight_;
  // Map the landmark ID to the expected observation.
  std::vector<LandmarkObs> expected_;
};

class ParticleFilter {
 public:
  // Constructor
  // @param num_particles Number of particles
  ParticleFilter() : initialized_(false) {}

  // Destructor
  ~ParticleFilter() {}

  /**
   * init Initializes particle filter by initializing particles to Gaussian
   *   distribution around first position and all the weights to 1.
   * @param x Initial x position [m] (simulated estimate from GPS)
   * @param y Initial y position [m]
   * @param theta Initial orientation [rad]
   * @param std[] Array of dimension 3 [standard deviation of x [m],
   *   standard deviation of y [m], standard deviation of yaw [rad]]
   */
  void init(unsigned int num_particles, double est_x, double est_y,
            double est_theta, double est_stddev[], double meas_stddev[],
            double sensor_range, const Map& map);

  /**
   * Update each particle's state prediction with process model.  Then,
   * update the list of landmarks expected to observe from that new position.
   * @param delta_t Time between time step t and t+1 in measurements [s]
   * @param std_pos[] Array of dimension 3 [standard deviation of x [m],
   *   standard deviation of y [m], standard deviation of yaw [rad]]
   * @param velocity Velocity of car from t to t+1 [m/s]
   * @param yaw_rate Yaw rate of car from t to t+1 [rad/s]
   */
  void prediction(double delta_t, double std_pos[], double velocity,
                  double yaw_rate);

  /**
   * dataAssociation Finds which observations correspond to which landmarks
   *   (likely by using a nearest-neighbors data association).
   * @param predicted Vector of predicted landmark observations
   * @param observations Vector of landmark observations
   */
  // void dataAssociation(std::vector<LandmarkObs> predicted,
  // std::vector<LandmarkObs>& observations);

  /**
   * updateWeights Updates the weights for each particle based on the likelihood
   *   of the observed measurements.
   * @param observations Vector of landmark observations
   */
  void updateWeights(std::vector<LandmarkObs> observations);

  /**
   * resample Resamples from the updated set of particles to form
   *   the new set of particles.
   */
  void resample();

  /**
   * initialized Returns whether particle filter is initialized yet or not.
   */
  bool initialized() const { return initialized_; }

  /**
   * Used for obtaining debugging information related to particles.
   */
  std::string getAssociations(Particle best);
  std::string getSenseCoord(Particle best, std::string coord);

  const std::vector<Particle>& particles() { return particles_; }

 private:
  /**
   * For each particle, compute the landmarks that are in sensor range.
   */
  void update_expected();
  const Map& map() const { return map_; }
  double sensor_range() const { return sensor_range_; }
  double gaussian_norm() const { return gaussian_norm_; }
  double norm_x() const { return norm_x_; }
  double norm_y() const { return norm_y_; }

  double gaussian_norm_;
  double norm_x_;
  double norm_y_;
  // Flag, if filter is initialized
  bool initialized_;

  // Set of current particles
  std::vector<Particle> particles_;

  double sensor_range_;
  Map map_;
};

#endif  // PARTICLE_FILTER_H_
