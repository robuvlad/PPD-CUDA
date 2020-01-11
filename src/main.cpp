#include <iostream>
#include "config.h"
#include <device_launch_parameters.h>

// OpenGL Graphics includes
#include <helper_gl.h>
#include <GL/freeglut.h>

// CUDA includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA utilities and system includes
#include <helper_cuda.h>
#include <helper_functions.h>
#include <rendercheck_gl.h>

#include <thread>
#include <math.h>
#include <chrono>


#include "model/Ant.h"
#include "model/FoodPack.h"

#include <mutex>

std::mutex mtx;


// =========== DATA SEGMENT =============
unsigned int antNumber = 100;
double gameSpeed = 0.001;
unsigned int foodPacksNumber = 5;
unsigned int avgPerFoodPack = 3;

Ant* ants;
FoodPack* foodPacks;
Position* anthillPos;
double* directionDeviations;

bool dataChanged = true;


double fRand(double fMin, double fMax)
{
	double f = (double)rand() / RAND_MAX;
	return fMin + f * (fMax - fMin);
}

void initializeState() {
	printf("Initializing game state\n");

	printf("Allocating memory for %d ants\n", antNumber);
	cudaMallocManaged(&ants, antNumber * sizeof(Ant));

	printf("Allocating memory for %d foodPacks\n", foodPacksNumber);
	cudaMallocManaged(&foodPacks, foodPacksNumber * sizeof(FoodPack));

	printf("Allocationg memory for direction deviations\n");
	cudaMallocManaged(&directionDeviations, antNumber * sizeof(double));

	cudaMallocManaged(&anthillPos, sizeof(Position));
	anthillPos = new Position(0.0f, 0.0f);

	printf("Initializing ants to starting state\n");
	for (unsigned int i = 0; i < antNumber; i++) {
		ants[i] = Ant(i, *anthillPos, gameSpeed, fRand(0, 359));
	}

	printf("Initializing food packs to starting state\n");
	for (unsigned int i = 0; i < foodPacksNumber; i++) {
		foodPacks[i] = FoodPack(avgPerFoodPack, Position(fRand(-1, 1), fRand(-1, 1)));
	}
}

// ==================== OPEN GL =====================================

void drawAnt(Ant* ant) {
	static const double antSize = 0.02;
	glBegin(GL_QUADS);                       
	glColor3f(1.0f, 0.0f, 0.0f);            
	glVertex2f(ant->position.x_pos + antSize, ant->position.y_pos);
	glVertex2f(ant->position.x_pos, ant->position.y_pos + antSize);
	glVertex2f(ant->position.x_pos - antSize, ant->position.y_pos);
	glVertex2f(ant->position.x_pos, ant->position.y_pos - antSize);
	glEnd();
}

void drawAnthill() {
	static const double anthillSize = 0.075;
	glBegin(GL_QUADS);
	glColor3f(1.0f, 0.5f, 0.0f);
	glVertex2f(anthillPos->x_pos + anthillSize, anthillPos->y_pos);
	glVertex2f(anthillPos->x_pos, anthillPos->y_pos + anthillSize);
	glVertex2f(anthillPos->x_pos - anthillSize, anthillPos->y_pos);
	glVertex2f(anthillPos->x_pos, anthillPos->y_pos - anthillSize);
	glEnd();
}

void drawFoodPack(FoodPack* foodPack) {
	static const double foodPackSize = 0.035;
	if (foodPack->food_amount > 0) {
		glBegin(GL_QUADS);
		glColor3f(0.0f, 0.8f, 0.0f);
		glVertex2f(foodPack->position.x_pos + foodPackSize, foodPack->position.y_pos);
		glVertex2f(foodPack->position.x_pos, foodPack->position.y_pos + foodPackSize);
		glVertex2f(foodPack->position.x_pos - foodPackSize, foodPack->position.y_pos);
		glVertex2f(foodPack->position.x_pos, foodPack->position.y_pos - foodPackSize);
		glEnd();
	}
}

/* Handler for window-repaint event. Call back when the window first appears and
whenever the window needs to be re-painted. */
void display() 
{
	if (dataChanged) {
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);    // Set background color to black and opaque
		glClear(GL_COLOR_BUFFER_BIT);            // Clear the color buffer (background)

		drawAnthill();

		for (unsigned int i = 0; i < antNumber; i++) {
			drawAnt(ants + i);
		}

		for (unsigned int i = 0; i < foodPacksNumber; i++) {
			drawFoodPack(foodPacks + i);
		}

		glFlush(); // Render now
		dataChanged = false;
	}
}


void runProgram(int argc, char **argv)
{
	glutInit(&argc, argv);      
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGTH);                 // Set the window's initial width & height
	glutInitWindowPosition(WINDOW_X_POSITION, WINDOW_Y_POSITION);    // Position the window's initial top-left corner
	glutCreateWindow(WINDOW_TITLE);                                  // Create a window with the given title
	glutDisplayFunc(display);                                        // Register display callback handler for window re-paint
	glutIdleFunc(display);                                     // Initialize GLUT
	glutMainLoop();                                                  // Enter the event-processing loop
}

// ======================== CUDA CONTEXT ================================

__global__ void moveAnt(Ant* ant, int antNumber, double gameSpeed, double* directionDeviation) { //global -> tells the compiler that this function will be executed on the gpu
	int i = threadIdx.x;

	if (i >= antNumber) {
		return;
	}

#ifdef DEBUG
	if (i == 2) {
		printf("ANT NR %d GAME SPEED %d\n", antNumber, gameSpeed);
		printf("POS ANT %f %f\n", (ant + i)->position.x_pos, (ant + i)->position.y_pos);
	}
#endif
	
	double x_move = gameSpeed * sin((ant + i)->direction);
	double y_move = gameSpeed * cos((ant + i)->direction);

	double new_x = (ant + i)->position.x_pos + x_move;
	double new_y = (ant + i)->position.y_pos + y_move;

	(ant + i)->position.x_pos = new_x;
	(ant + i)->position.y_pos = new_y;

	(ant + i)->direction += directionDeviation[i];
}

void cudaThread() {
	printf("Running CUDA thread\n");
	

	while (true) {

		for (unsigned int i = 0; i < antNumber; i++) {
			directionDeviations[i] = fRand(-0.2f, 0.2f);
		}

		moveAnt <<<1, antNumber>>> (ants, antNumber, gameSpeed, directionDeviations);

		cudaDeviceSynchronize();

		dataChanged = true;

		std::this_thread::sleep_for(std::chrono::milliseconds(30));
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