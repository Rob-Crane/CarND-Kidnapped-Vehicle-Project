#ifndef KIDNAPPED_VEHICLE_TYPES_H_
#define KIDNAPPED_VEHICLE_TYPES_H_

#include <vector>

class CartesianPoint {
public:
  CartesianPoint(double x, double y) : x_(x), y_(y) {}
  double x() const { return x_; }
  double y() const { return y_; }
protected:
  double x_;
  double y_;
};

/**
 * Computes the Euclidean distance between two 2D points.
 * @param point1 first point
 * @param point2 econd point
 * @output Euclidean distance between two 2D points
 */
inline double dist(const CartesianPoint& point1, const CartesianPoint point2) {
  double x_dist = point1.x() - point2.x();
  double y_dist = point1.y() - point2.y();
  return sqrt(x_dist * x_dist + y_dist * y_dist);
}


/**
 * Struct representing one landmark observation measurement.
 */
class LandmarkObs : public CartesianPoint {
public:
  LandmarkObs(double x, double y) : CartesianPoint(x, y) {}
  void set_landmark(unsigned int landmark_id) { landmark_id_ = landmark_id; }
  unsigned int landmark_id() const { return landmark_id_; }
private: 
  unsigned int landmark_id_;    // Id of matching landmark in the map.
};

/**
 * Struct representing one position/control measurement.
 */
struct control_s {
  double velocity;  // Velocity [m/s]
  double yawrate;   // Yaw rate [rad/s]
};

/**
 * Struct representing one ground truth position.
 */
struct ground_truth {
  double x;      // Global vehicle x position [m]
  double y;      // Global vehicle y position
  double theta;  // Global vehicle yaw [rad]
};

class Landmark : public CartesianPoint {
public:
  Landmark(double x, double y, unsigned int id) : CartesianPoint(x, y), id_(id) {}
  unsigned int id() const { return id_; }
private:
  unsigned int id_; // Landmark ID
};

struct Map {
  std::vector<Landmark> landmark_list; // List of landmarks in the map
};

#endif  // KIDNAPPED_VEHICLE_TYPES_H_
