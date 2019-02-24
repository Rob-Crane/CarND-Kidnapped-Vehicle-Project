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

void Particle::update_expected(vector<LandmarkObs>&& expected) {
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
    vector<LandmarkObs> expected;
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
    if(fabs(yaw_rate) < 1E-5) {
        double x = p.x() + velocity * delta_t * cos(p.theta());
        double y = p.y() + velocity * delta_t * sin(p.theta());
        p.update_estimate(x, y, p.theta());
    } else {
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
  }
  update_expected();
}
#include <iostream>
void ParticleFilter::updateWeights(vector<LandmarkObs> observations) {
  for (Particle& p : particles_) {
    double weight = 1.0;
    // Compute weight as product of probabilities of observing each expected
    // landmark.
    for (const LandmarkObs& expected_obs : p.expected()) {
      // Find the observation that is closest to expected landmark observation.
      vector<LandmarkObs>::const_iterator closest_obs;
      double min_dist = std::numeric_limits<double>::max();
      for (vector<LandmarkObs>::const_iterator obs_it = observations.cbegin();
           obs_it != observations.cend(); ++obs_it) {
        // Transform observation to global frame.
        double obs_x = p.x() + (obs_it->x() * cos(p.theta()) -
                                obs_it->y() * sin(p.theta()));
        double obs_y = p.y() + (obs_it->x() * sin(p.theta()) +
                                obs_it->y() * cos(p.theta()));
        std::cout<<"x: "<< obs_x << " y: " << obs_y << std::endl;
        std::exit(0);
        const LandmarkObs transformed_obs(obs_x, obs_y);
        double distance = dist(expected_obs, transformed_obs);
        if (distance < min_dist) {
          min_dist = distance;
          closest_obs = obs_it;
        }
      }
      // Calculate weight from M.V. Gaussian.
      double diff_x = closest_obs->x() - expected_obs.x();
      double diff_y = closest_obs->y() - expected_obs.y();
      double exponent =
          -(norm_x() * diff_x * diff_x + norm_y() * diff_y * diff_y);
      weight *= (gaussian_norm() * exp(exponent));
    }
    p.update_weight(weight);
  }
}

void ParticleFilter::resample() {
  vector<double> weights;
  for (const Particle& p : particles()) {
    weights.push_back(p.weight());
  }
  std::discrete_distribution<unsigned int> weights_dist(weights.cbegin(),
                                                        weights.cend());
  std::default_random_engine gen;
  vector<Particle> resampled_particles;
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
