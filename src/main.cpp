#include <iostream>
#include "config.h"

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


/* Handler for window-repaint event. Call back when the window first appears and
whenever the window needs to be re-painted. */
void display() 
{
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);    // Set background color to black and opaque
	glClear(GL_COLOR_BUFFER_BIT);            // Clear the color buffer (background)
	glBegin(GL_QUADS);                       // Each set of 4 vertices form a quad
	glColor3f(1.0f, 0.0f, 0.0f);             // Red
	glVertex2f(-0.5f, -0.5f);                // x, y
	glVertex2f(0.5f, -0.5f);
	glVertex2f(0.5f, 0.5f);
	glVertex2f(-0.5f, 0.5f);
	glEnd();
	glFlush();                               // Render now
}


void runProgram(int argc, char **argv)
{
	glutInit(&argc, argv);                                           // Initialize GLUT
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGTH);                 // Set the window's initial width & height
	glutInitWindowPosition(WINDOW_X_POSITION, WINDOW_Y_POSITION);    // Position the window's initial top-left corner
	glutCreateWindow(WINDOW_TITLE);                                  // Create a window with the given title
	glutDisplayFunc(display);                                        // Register display callback handler for window re-paint
	glutMainLoop();                                                  // Enter the event-processing loop
}


int main(int argc, char **argv) 
{
	std::cout << "Starting ANT CUDA application" << std::endl;

	runProgram(argc, argv);

	std::cout << "Closing ANT CUDA application :(" << std::endl;
	return 0;
}