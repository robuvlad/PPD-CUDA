#pragma once


class Ant
{
public:
	int index;
	double x_pos;
	double y_pos;
	double speed;
	double direction;
	bool has_food;

	Ant(int index, double x_pos, double y_pos, double speed, double direction);

	~Ant();
};
