/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
//#include <algorithm>
#include <iterator>
//#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::sin;
using std::exp;
using std::cos;
using std::sqrt;
using std::string;
using std::vector;

Particle::Particle(double x, double y, double theta)
    : CartesianPoint(x, y), theta_(theta), weight_(1.0) {}

void Particle::update_estimate(double x, double y, double theta) {
  x_ = x;
  y_ = y;
  theta_ = theta;
}

void Particle::update_weight(double weight) { weight_ = weight; }

void Particle::update_expected(std::vector<LandmarkObs>&& expected) {
  expected_ = expected;
}

vector<int> Particle::associations() {
  vector<int> associations;
  for (const LandmarkObs& expected_obs : expected()) {
    associations.push_back(expected_obs.landmark_id());
  }
  return std::move(associations);
}

vector<double> Particle::sense_x() {
  vector<double> sense_x;
  for (const LandmarkObs& expected_obs : expected()) {
    sense_x.push_back(expected_obs.x());
  }
  return std::move(sense_x);
}

vector<double> Particle::sense_y() {
  vector<double> sense_y;
  for (const LandmarkObs& expected_obs : expected()) {
    sense_y.push_back(expected_obs.y());
  }
  return std::move(sense_y);
}

void ParticleFilter::update_expected() {
  for (Particle& p : particles_) {
    std::vector<LandmarkObs> expected;
    for (const auto& landmark : map().landmark_list) {
      double distance = dist(landmark, p);
      if (distance < sensor_range()) {
        LandmarkObs pred_obs(landmark.x(), landmark.y());
        pred_obs.set_landmark(landmark.id());
        expected.push_back(pred_obs);
      }
    }
    p.update_expected(std::move(expected));
  }
}

void ParticleFilter::init(unsigned int num_particles, double est_x,
                          double est_y, double est_theta, double est_stddev[],
                          double meas_stddev[], double sensor_range,
                          const Map& map) {
  map_ = map;
  sensor_range_ = sensor_range;
  gaussian_norm_ = 1.0 / (2.0 * M_PI * meas_stddev[0] * meas_stddev[1]);
  norm_x_ = 1.0 / (2.0 * meas_stddev[0] * meas_stddev[0]);
  norm_y_ = 1.0 / (2.0 * meas_stddev[1] * meas_stddev[1]);

  // Initialize particles to default
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(est_x, est_stddev[0]);
  std::normal_distribution<double> dist_y(est_y, est_stddev[1]);
  std::normal_distribution<double> dist_theta(est_theta, est_stddev[2]);
  for (unsigned int i = 0; i < num_particles; ++i) {
    double x = dist_x(gen);
    double y = dist_y(gen);
    double theta = dist_theta(gen);
    particles_.emplace_back(x, y, theta);
  }
  update_expected();
  initialized_ = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  std::default_random_engine gen;
  std::normal_distribution<double> dist_noise_x(0.0, std_pos[0]);
  std::normal_distribution<double> dist_noise_y(0.0, std_pos[1]);
  std::normal_distribution<double> dist_noise_theta(0.0, std_pos[2]);

  for (Particle& p : particles_) {
    double x = p.x() +
               velocity / yaw_rate *
                   (sin(p.theta() + yaw_rate * delta_t) - sin(p.theta())) +
               dist_noise_x(gen);
    double y = p.y() +
               velocity / yaw_rate *
                   (cos(p.theta()) - cos(p.theta() + yaw_rate * delta_t)) +
               dist_noise_y(gen);
    double theta = p.theta() + yaw_rate * delta_t + dist_noise_theta(gen);
    p.update_estimate(x, y, theta);
  }
  update_expected();
}

#include <iostream>
void ParticleFilter::updateWeights(vector<LandmarkObs> observations) {
  for (Particle& p : particles_) {
    double weight = 1.0;
    for (const LandmarkObs& obs : observations) {
      // Get closest expect landmark observation for this particle.
      double min_dist = std::numeric_limits<double>::max();
      std::vector<LandmarkObs>::const_iterator closest_expected;
      for (std::vector<LandmarkObs>::const_iterator exp_it =
               p.expected().cbegin();
           exp_it != p.expected().cend(); ++exp_it) {
        double distance = dist(obs, *exp_it);
        if (distance < min_dist) {
          min_dist = distance;
          closest_expected = exp_it;
        }
      }
      // Transform observation to global frame.
      double obs_x =
          p.x() + (obs.x() * cos(p.theta()) - obs.y() * sin(p.theta()));
      double obs_y =
          p.y() + (obs.x() * sin(p.theta()) - obs.y() * cos(p.theta()));
      // Calculate weight from M.V. Gaussian.
      double diff_x = closest_expected->x() - obs_x;
      double diff_y = closest_expected->y() - obs_y;
      double exponent =
          -(norm_x() * diff_x * diff_x + norm_y() * diff_y * diff_y);
      //TODO find out reason this weight is 0 always
      weight *= (gaussian_norm() * exp(exponent));
    }
    std::cout<<"calc weight: "<<weight<<std::endl;
    p.update_weight(weight);
  }
}

void ParticleFilter::resample() {
  std::vector<double> weights;
  for (const Particle& p : particles()) {
    weights.push_back(p.weight());
  }
  std::discrete_distribution<unsigned int> weights_dist(weights.cbegin(),
                                                        weights.cend());
  std::default_random_engine gen;
  std::vector<Particle> resampled_particles;
  for (unsigned int i = 0; i < particles().size(); ++i) {
    unsigned int j = weights_dist(gen);
    resampled_particles.push_back(particles()[j]);
  }
  particles_ = std::move(resampled_particles);
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations();
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x();
  } else {
    v = best.sense_y();
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
