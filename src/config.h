/* Configuration file for the whole project */
#pragma once

// Business logic configurations
#define ANTS_AMOUNT             70                 // The number of ants that will be generated
#define NUMBER_FOOD_PACKS       100                  // The number of food packs that will be placed on the map
#define AVG_FOOD_IN_PACK        40                  // The average amount of food that will be found a food pack
#define GAME_SPEED              0.005                // The speed at which the ants will move (normal speed is 1.0)
#define PROXIMITY_RADIUS        0.04

// OpenGL configurations
#define WINDOW_WIDTH            800
#define WINDOW_HEIGTH           800
#define WINDOW_X_POSITION       300
#define WINDOW_Y_POSITION       200
#define WINDOW_TITLE            "ANT Simulation"
#define ANT_SIZE				0.02
#define ANTHILL_SIZE			0.075
#define FOOD_PACK_SIZE			0.015

// Other
#define PI						3.14159265
#define THREAD_SLEEP			60
