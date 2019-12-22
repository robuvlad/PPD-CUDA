#include "Ant.h"


Ant::Ant(int index, double x_pos, double y_pos, double speed, double direction) :
	index{ index }, x_pos{ x_pos }, y_pos{ y_pos }, speed{ speed }, direction{ direction }, has_food{ false } {
}


Ant::~Ant()
{
}
