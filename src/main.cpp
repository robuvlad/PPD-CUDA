#include <iostream>
#include <device_launch_parameters.h>
#include <helper_gl.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <rendercheck_gl.h>
#include <thread>
#include <math.h>
#include <chrono>
#include "config.h"
#include "model/Ant.h"
#include "model/FoodPack.h"


// =========== DATA SEGMENT =============

Ant* ants;
FoodPack* foodPacks;
Position* anthillPos;
double* directionDeviations;
bool dataChanged = true;

// =========== UTILS ===========

double fRand(double fMin, double fMax)
{
	double f = (double)rand() / RAND_MAX;
	return fMin + f * (fMax - fMin);
}

void initializeState() {
	printf("Initializing game state\n");

	printf("Allocating memory for %d ants\n", ANTS_AMOUNT);
	cudaMallocManaged(&ants, ANTS_AMOUNT * sizeof(Ant));

	printf("Allocating memory for %d foodPacks\n", NUMBER_FOOD_PACKS);
	cudaMallocManaged(&foodPacks, NUMBER_FOOD_PACKS * sizeof(FoodPack));

	printf("Allocationg memory for direction deviations\n");
	cudaMallocManaged(&directionDeviations, ANTS_AMOUNT * sizeof(double));

	cudaMallocManaged(&anthillPos, sizeof(Position));
	anthillPos = new Position(0.0f, 0.0f);

	printf("Initializing ants to starting state\n");
	for (unsigned int i = 0; i < ANTS_AMOUNT; i++) {
		ants[i] = Ant(i, *anthillPos, GAME_SPEED, fRand(0, 359));
	}

	printf("Initializing food packs to starting state\n");
	for (unsigned int i = 0; i < NUMBER_FOOD_PACKS; i++) {
		double xPos = fRand(-1, 1);
		double yPos = fRand(-1, 1);
		if (yPos < 0.3) {
			yPos += 0.3;
		}
		else if (yPos > -0.3 && yPos < 0) {
			yPos -= 0.3;
		}
		foodPacks[i] = FoodPack(AVG_FOOD_IN_PACK, Position(xPos, yPos));
	}
}

// =========== OPEN GL ===========

void drawAnt(Ant* ant) {
	glBegin(GL_QUADS);
	if (ant->has_food) {
		glColor3f(1.0f, 3.0f, 2.0f);
	}
	else {
		glColor3f(1.0f, 0.0f, 0.0f);
	}
	glVertex2f(ant->position.x_pos + ANT_SIZE, ant->position.y_pos);
	glVertex2f(ant->position.x_pos, ant->position.y_pos + ANT_SIZE);
	glVertex2f(ant->position.x_pos - ANT_SIZE, ant->position.y_pos);
	glVertex2f(ant->position.x_pos, ant->position.y_pos - ANT_SIZE);
	glEnd();
}

void drawAnthill() {
	glBegin(GL_QUADS);
	glColor3f(1.0f, 0.5f, 0.0f);
	glVertex2f(anthillPos->x_pos + ANTHILL_SIZE, anthillPos->y_pos);
	glVertex2f(anthillPos->x_pos, anthillPos->y_pos + ANTHILL_SIZE);
	glVertex2f(anthillPos->x_pos - ANTHILL_SIZE, anthillPos->y_pos);
	glVertex2f(anthillPos->x_pos, anthillPos->y_pos - ANTHILL_SIZE);
	glEnd();
}

void drawFoodPack(FoodPack* foodPack) {
	double foodPackSize = FOOD_PACK_SIZE * (foodPack->food_amount > 0 ? log(foodPack->food_amount) + 1 : 1);
	glBegin(GL_TRIANGLES);
	if (foodPack->food_amount > 0) {
		glColor3f(0.0f, 0.8f, 0.0f);
	}
	else {
		glColor3f(0.4f, 0.2f, 0.5f);
	}
	glVertex2f(foodPack->position.x_pos, foodPack->position.y_pos + foodPackSize);
	glVertex2f(foodPack->position.x_pos + foodPackSize, foodPack->position.y_pos - foodPackSize);
	glVertex2f(foodPack->position.x_pos - foodPackSize, foodPack->position.y_pos - foodPackSize);
	glEnd();
}

void display() 
{
	if (dataChanged) {
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		drawAnthill();

		for (unsigned int i = 0; i < ANTS_AMOUNT; i++) {
			drawAnt(ants + i);
		}

		for (unsigned int i = 0; i < NUMBER_FOOD_PACKS; i++) {
			drawFoodPack(foodPacks + i);
		}

		glFlush();
		dataChanged = false;
	}
}


void runProgram(int argc, char **argv)
{
	glutInit(&argc, argv);      
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGTH);
	glutInitWindowPosition(WINDOW_X_POSITION, WINDOW_Y_POSITION);
	glutCreateWindow(WINDOW_TITLE);
	glutDisplayFunc(display);
	glutIdleFunc(display);
	glutMainLoop();
}

// ======================== CUDA CONTEXT ================================

__global__ void moveAnt(Ant* ant, int antNumber, double gameSpeed, double* directionDeviation, FoodPack* foodPacks, int foodPacksNumber, double radius) {
	int i = threadIdx.x;

	if (i >= antNumber) {
		return;
	}

	if ((ant + i)->has_food) {
		double distanceToAnthill = sqrt(((ant + i)->position.x_pos * (ant + i)->position.x_pos) + ((ant + i)->position.y_pos * (ant + i)->position.y_pos));
		if (distanceToAnthill <= radius) {
			(ant + i)->has_food = false;
			(ant + i)->direction = (ant + i)->direction < 180 ? (ant + i)->direction + 180 : (ant + i)->direction - 180;
		}
		else {
			double radDirection = (ant + i)->direction * PI / 180;

			double x_move = gameSpeed * cos(radDirection);
			double y_move = gameSpeed * sin(radDirection);

			double new_x = (ant + i)->position.x_pos - x_move;
			double new_y = (ant + i)->position.y_pos - y_move;

			(ant + i)->position.x_pos = new_x;
			(ant + i)->position.y_pos = new_y;
		}
	}
	else {
		bool isNearFoodPack = false;

		for (unsigned int foodPackIndex = 0; foodPackIndex < foodPacksNumber; foodPackIndex++) {
			double xDiff = (ant + i)->position.x_pos - foodPacks[foodPackIndex].position.x_pos;
			double yDiff = (ant + i)->position.y_pos - foodPacks[foodPackIndex].position.y_pos;
			double distanceAntFoodPack = sqrt(xDiff * xDiff + yDiff * yDiff);


			if (distanceAntFoodPack <= radius && foodPacks[foodPackIndex].food_amount > 0) {
				printf("ANT nr %d found food pack!!!\n", i);
				foodPacks[foodPackIndex].food_amount -= 1;
				ant[i].has_food = true;
				isNearFoodPack = true;
				break;
			}
		}


		if (isNearFoodPack) {
			double distanceToAnthill = sqrt(((ant + i)->position.x_pos * (ant + i)->position.x_pos) + ((ant + i)->position.y_pos * (ant + i)->position.y_pos));
			ant[i].direction = acos(abs(ant[i].position.x_pos) / distanceToAnthill) * 180.0 / PI;
			if ((ant + i)->position.x_pos <= 0 && (ant + i)->position.y_pos >= 0) {
				ant[i].direction = 180 - ant[i].direction;
			}
			else if ((ant + i)->position.x_pos <= 0 && (ant + i)->position.y_pos <= 0) {
				ant[i].direction = 180 + ant[i].direction;
			} else if ((ant + i)->position.x_pos >= 0 && (ant + i)->position.y_pos <= 0) {
				ant[i].direction = 360 - ant[i].direction;
			}
		}
		else {
			double x_move = gameSpeed * sin((ant + i)->direction);
			double y_move = gameSpeed * cos((ant + i)->direction);

			double new_x = (ant + i)->position.x_pos - x_move;
			double new_y = (ant + i)->position.y_pos - y_move;

			(ant + i)->position.x_pos = new_x;
			(ant + i)->position.y_pos = new_y;

			(ant + i)->direction += directionDeviation[i];

			if (abs((ant + i)->position.x_pos) >= 1 || abs((ant + i)->position.y_pos) >= 1) {
				(ant + i)->direction += 180;
			}
		}
	}
}

void cudaThread() {
	printf("Running CUDA thread\n");

	while (true) {

		for (unsigned int i = 0; i < ANTS_AMOUNT; i++) {
			directionDeviations[i] = fRand(-0.2f, 0.2f);
		}

		moveAnt <<<1, ANTS_AMOUNT>>> (ants, ANTS_AMOUNT, GAME_SPEED, directionDeviations, foodPacks, NUMBER_FOOD_PACKS, PROXIMITY_RADIUS);

		cudaDeviceSynchronize();

		dataChanged = true;

		std::this_thread::sleep_for(std::chrono::milliseconds(THREAD_SLEEP));
	}
}

// =================== APP ENTRY POINT =======================

int main(int argc, char **argv) 
{
	srand(time(NULL));

	std::cout << "Starting ANT CUDA application" << std::endl;

	initializeState();
	std::thread thread_obj(cudaThread);
	runProgram(argc, argv);

	std::cout << "Closing ANT CUDA application :(" << std::endl;
	return 0;
}
