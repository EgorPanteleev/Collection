#include "Camera.h"
Camera::Camera(): origin(), dV(0), Vx(0), Vy(0) {
}
Camera::Camera( Point pos, double dv, double vx, double vy ): origin( pos ), dV(dv), Vx(vx), Vy(vy) {
}