#include "Ant.h"


Ant::Ant(const unsigned int index, const Position position, const double speed, const double direction) :
	index{ index }, position{ position }, speed{ speed }, direction{ direction }, has_food{ false } {
}


Ant::~Ant()
{
}
