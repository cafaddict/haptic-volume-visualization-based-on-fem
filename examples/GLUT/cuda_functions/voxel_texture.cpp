//==============================================================================
/*
	Software License Agreement (BSD License)
	Copyright (c) 2003-2016, CHAI3D.
	(www.chai3d.org)

	All rights reserved.

	Redistribution and use in source and binary forms, with or without
	modification, are permitted provided that the following conditions
	are met:

	* Redistributions of source code must retain the above copyright
	notice, this list of conditions and the following disclaimer.

	* Redistributions in binary form must reproduce the above
	copyright notice, this list of conditions and the following
	disclaimer in the documentation and/or other materials provided
	with the distribution.

	* Neither the name of CHAI3D nor the names of its contributors may
	be used to endorse or promote products derived from this software
	without specific prior written permission.

	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
	"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
	LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
	FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
	COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
	INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
	BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
	LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
	CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
	LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
	ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
	POSSIBILITY OF SUCH DAMAGE.

	\author    <http://www.chai3d.org>
	\author    Sonny Chan
	\author    Francois Conti
	\version   3.1.1 $Rev: 1292 $
*/
//==============================================================================

//------------------------------------------------------------------------------
#include "chai3d.h"
#include "cuda_header.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <ctime>
//------------------------------------------------------------------------------
using namespace chai3d;
using namespace std;
//------------------------------------------------------------------------------
#ifndef MACOSX
#include "GL/glut.h"
#else
#include "GLUT/glut.h"
#endif
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// GENERAL SETTINGS
//------------------------------------------------------------------------------

// stereo Mode
/*
	C_STEREO_DISABLED:            Stereo is disabled
	C_STEREO_ACTIVE:              Active stereo for OpenGL NVDIA QUADRO cards
	C_STEREO_PASSIVE_LEFT_RIGHT:  Passive stereo where L/R images are rendered next to each other
	C_STEREO_PASSIVE_TOP_BOTTOM:  Passive stereo where L/R images are rendered above each other
*/
cStereoMode stereoMode = C_STEREO_DISABLED;

// fullscreen mode
bool fullscreen = false;

// mirrored display
bool mirroredDisplay = false;

// Mutex to voxel
cMutex mutexVoxel;

//------------------------------------------------------------------------------
// DECLARED VARIABLES
//------------------------------------------------------------------------------

// a world that contains all objects of the virtual environment
cWorld* world;

// a camera to render the world in the window display
cCamera* camera;

// a light source to illuminate the objects in the world
cDirectionalLight *light;

// a haptic device handler
cHapticDeviceHandler* handler;

// a pointer to the current haptic device
cGenericHapticDevicePtr hapticDevice;

// a virtual tool representing the haptic device in the scene
cToolCursor* tool;

// a sphere to show the projected point on the surface
cShapeSphere* cursor;

// a virtual hand object from a CT scan
cVoxelObject* object;

// angular velocity of object
cVector3d angVel(0.0, 0.0, 0.1);

// a label to display the rate [Hz] at which the simulation is running
cLabel* labelHapticRate;

// a label to display the rate [Hz] at which the graphic is running
cLabel* labelGraphicRate;

// a label to explain what is happening
cLabel* labelMessage;

// indicates if the haptic simulation currently running
bool simulationRunning = false;

// indicates if the haptic simulation has terminated
bool simulationFinished = true;

// frequency counter to measure the simulation haptic rate
cFrequencyCounter frequencyCounter;


// frequency counter to measure the simulation graphic rate
cFrequencyCounter frequencyCounter_graphic;

// last mouse position
int mouseX;
int mouseY;

// information about computer screen and GLUT display window
int screenW;
int screenH;
int windowW;
int windowH;
int windowPosX;
int windowPosY;

double maxStiffness;
//------------------------------------------------------------------------------
// DECLARED MACROS
//------------------------------------------------------------------------------
// convert to resource path
#define RESOURCE_PATH(p)    (char*)((resourceRoot+string(p)).c_str())


//------------------------------------------------------------------------------
// DECLARED FUNCTIONS
//------------------------------------------------------------------------------

// callback when the window display is resized
void resizeWindow(int w, int h);

// callback when a key from the representing is pressed
void keySelect(unsigned char key, int x, int y);

// callback to handle mouse click
void mouseClick(int button, int state, int x, int y);

// callback to handle mouse motion when button is pressed
void mouseMove(int x, int y);

// function called before exiting the application
void close(void);

// callback to render graphic scene
void updateGraphics(void);

// callback of GLUT timer
void graphicsTimer(int data);

// main haptics loop
void updateHaptics(void);


// For my research
cVector3d prevpos;
double prevscale;

enum Material
{
	SKIN,
	BONE,
	BRAIN
};


Material materialTransferFunc(float intensity);
float defineProperty(cCollisionEvent* contact, cVoxelObject* object);

// Collision check
bool CollisionCheck(cVoxelObject* voxObject, cToolCursor* tool_cursor);
cVector3d cVec3Clamp(cVector3d &a_value, cVector3d &a_low, cVector3d &a_high);

float get_density(int x, int y, int z);
cVector3d get_gradient(int x, int y, int z);

cCollisionEvent* contact;
cMultiMesh* arrow;

int flag = 0;

bool zero_force_flag = true;
cVector3d zero_force_position;

cMultiImagePtr custom_image;
cTexture3dPtr custom_texture;

void buildVoxelCube();
cVector3d VoxIdx_to_GPos();
cVector3d VoxIdx_to_GPos2(int voxX, int voxY, int voxZ);

cVector3d arrowPos;
cVector3d arrowPrevPos;

cVector3d returnForce;

Material prev_mat;
Material cur_mat;

time_t start;
int debug_vox_pos;
int voxResolution = 256;

cImagePtr boneLUT;
cImagePtr softLUT;

struct Element
{
	void CalculateStiffnessMatrix(const Eigen::Matrix3f& D, std::vector<Eigen::Triplet<float>>& triplets);

	Eigen::Matrix<float, 3, 6> B;
	int nodesIds[3];
};

struct Constraint
{
	enum Type
	{
		UX = 1 << 0,
		UY = 1 << 1,
		UXY = UX | UY
	};
	int node;
	Type type;
};

int nodesCount;
Eigen::VectorXf nodesX;
Eigen::VectorXf nodesY;
Eigen::VectorXf loads;

std::vector<Element> elements;
std::vector<Constraint> constraints;
std::vector<int> constraint_nodes;


int hit_flag = 0;


void Element::CalculateStiffnessMatrix(const Eigen::Matrix3f& D, std::vector<Eigen::Triplet<float> >& triplets)
{
	
	Eigen::Vector3f x, y;
	x << nodesX[nodesIds[0]], nodesX[nodesIds[1]], nodesX[nodesIds[2]];
	y << nodesY[nodesIds[0]], nodesY[nodesIds[1]], nodesY[nodesIds[2]];

	Eigen::Matrix3f C;
	C << Eigen::Vector3f(1.0f, 1.0f, 1.0f), x, y;

	Eigen::Matrix3f IC = C.inverse();
	B.setZero();
	for (int i = 0; i < 3; i++)
	{
		B(0, 2 * i + 0) = IC(1, i);
		B(0, 2 * i + 1) = 0.0f;
		B(1, 2 * i + 0) = 0.0f;
		B(1, 2 * i + 1) = IC(2, i);
		B(2, 2 * i + 0) = IC(2, i);
		B(2, 2 * i + 1) = IC(1, i);
	}
	
	Eigen::Matrix<float, 6, 6> K = B.transpose() * D * B * C.determinant() / 2.0f;
	
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			Eigen::Triplet<float> trplt11(2 * nodesIds[i] + 0, 2 * nodesIds[j] + 0, K(2 * i + 0, 2 * j + 0));
			Eigen::Triplet<float> trplt12(2 * nodesIds[i] + 0, 2 * nodesIds[j] + 1, K(2 * i + 0, 2 * j + 1));
			Eigen::Triplet<float> trplt21(2 * nodesIds[i] + 1, 2 * nodesIds[j] + 0, K(2 * i + 1, 2 * j + 0));
			Eigen::Triplet<float> trplt22(2 * nodesIds[i] + 1, 2 * nodesIds[j] + 1, K(2 * i + 1, 2 * j + 1));

			triplets.push_back(trplt11);
			triplets.push_back(trplt12);
			triplets.push_back(trplt21);
			triplets.push_back(trplt22);
		}
	}
}

void SetConstraints(Eigen::SparseMatrix<float>::InnerIterator& it, int index)
{
	if (it.row() == index || it.col() == index)
	{
		it.valueRef() = it.row() == it.col() ? 1.0f : 0.0f;
	}
}

void ApplyConstraints(Eigen::SparseMatrix<float>& K, const std::vector<Constraint>& constraints)
{
	std::vector<int> indicesToConstraint;
	for (std::vector<Constraint>::const_iterator it = constraints.begin(); it != constraints.end(); ++it)
	{
		if (it->type & Constraint::UX)
		{
			indicesToConstraint.push_back(2 * it->node + 0);
		}
		if (it->type & Constraint::UY)
		{
			indicesToConstraint.push_back(2 * it->node + 1);
		}
	}

	for (int k = 0; k < K.outerSize(); ++k)
	{
		for (Eigen::SparseMatrix<float>::InnerIterator it(K, k); it; ++it)
		{
			for (std::vector<int>::iterator idit = indicesToConstraint.begin(); idit != indicesToConstraint.end(); ++idit)
			{
				SetConstraints(it, *idit);
			}
		}
	}
}


//==============================================================================
/*
	DEMO:    29-isosurface.cpp

	This demonstration illustrates the use of a isosurfaces.
*/
//==============================================================================
string filePath;
ofstream writeFile;

int main(int argc, char* argv[])
{
	filePath = "C:/Users/CGV/Desktop/data/test.txt";
	writeFile = ofstream(filePath.data());
	//--------------------------------------------------------------------------
	// INITIALIZATION
	//--------------------------------------------------------------------------
	cout << endl;
	cout << "-----------------------------------" << endl;
	cout << "CHAI3D" << endl;
	cout << "Demo: 29-isosurface" << endl;
	cout << "Copyright 2003-2016" << endl;
	cout << "-----------------------------------" << endl << endl << endl;
	cout << "Keyboard Options:" << endl << endl;
	cout << "[1] - Select first isosurface" << endl;
	cout << "[2] - Select second isosurface" << endl;
	cout << "[q,w] Adjust quality of graphic rendering" << endl;
	cout << "[f] - Enable/Disable full screen mode" << endl;
	cout << "[m] - Enable/Disable vertical mirroring" << endl;
	cout << "[x] - Exit application" << endl;
	cout << endl << endl;

	// parse first arg to try and locate resources
	string resourceRoot = string(argv[0]).substr(0, string(argv[0]).find_last_of("/\\") + 1);


	//--------------------------------------------------------------------------
	// OPEN GL - WINDOW DISPLAY
	//--------------------------------------------------------------------------

	// initialize GLUT
	glutInit(&argc, argv);

	// retrieve  resolution of computer display and position window accordingly
	screenW = glutGet(GLUT_SCREEN_WIDTH);
	screenH = glutGet(GLUT_SCREEN_HEIGHT);
	windowW = 0.8 * screenH;
	windowH = 0.5 * screenH;
	windowPosY = (screenH - windowH) / 2;
	windowPosX = windowPosY;

	// initialize the OpenGL GLUT window
	glutInitWindowPosition(windowPosX, windowPosY);
	glutInitWindowSize(windowW, windowH);

	if (stereoMode == C_STEREO_ACTIVE)
		glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE | GLUT_STEREO);
	else
		glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);

	// create display context and initialize GLEW library
	glutCreateWindow(argv[0]);

#ifdef GLEW_VERSION
	// initialize GLEW
	glewInit();
#endif

	// setup GLUT options
	glutDisplayFunc(updateGraphics);
	glutKeyboardFunc(keySelect);
	glutMouseFunc(mouseClick);
	glutMotionFunc(mouseMove);
	glutReshapeFunc(resizeWindow);
	glutSetWindowTitle("CHAI3D");

	// set fullscreen mode
	if (fullscreen)
	{
		glutFullScreen();
	}



	//--------------------------------------------------------------------------
	// WORLD - CAMERA - LIGHTING
	//--------------------------------------------------------------------------

	// create a new world.
	world = new cWorld();

	// set the background color of the environment
	world->m_backgroundColor.setBlack();

	// create a camera and insert it into the virtual world
	camera = new cCamera(world);
	world->addChild(camera);

	// define a basis in spherical coordinates for the camera
	
	/*uncomment below when using complex object*/
	//camera->setSphericalReferences(cVector3d(0, 0, 0),    // origin
	//	cVector3d(0, 0, 1),    // zenith direction
	//	cVector3d(1, 0, 0));   // azimuth direction

	//camera->setSphericalDeg(1.2,    // spherical coordinate radius
	//	30,     // spherical coordinate azimuth angle
	//	-10);    // spherical coordinate polar angle

	camera->setSphericalReferences(cVector3d(0, 0, 0),    // origin
		cVector3d(0, 0, 1),    // zenith direction
		cVector3d(1, 0, 0));   // azimuth direction

	camera->setSphericalDeg(2.3,    // spherical coordinate radius
		30,     // spherical coordinate azimuth angle
		10);    // spherical coordinate polar angle



	//camera->setSphericalDeg(2.3,    // spherical coordinate radius
	//	0,     // spherical coordinate azimuth angle
	//	0);    // spherical coordinate polar angle

// set the near and far clipping planes of the camera
// anything in front or behind these clipping planes will not be rendered
	camera->setClippingPlanes(0.1, 10.0);

	// set stereo mode
	camera->setStereoMode(stereoMode);

	// set stereo eye separation and focal length (applies only if stereo is enabled)
	//camera->setStereoEyeSeparation(0.02);
	//camera->setStereoFocalLength(2.0);

	camera->setStereoEyeSeparation(0.03);
	camera->setStereoFocalLength(3.0);

	// set vertical mirrored display mode
	camera->setMirrorVertical(mirroredDisplay);

	// create a light source
	light = new cDirectionalLight(world);

	// attach light to camera
	camera->addChild(light);

	// enable light source
	light->setEnabled(true);

	// define the direction of the light beam
	light->setDir(-3.0, -0.5, 0.0);
	//light->setDir(0.0, 0.0, 0.0);


	//--------------------------------------------------------------------------
	// HAPTIC DEVICES / TOOLS
	//--------------------------------------------------------------------------

	// create a haptic device handler
	handler = new cHapticDeviceHandler();

	// get access to the first available haptic device found
	handler->getDevice(hapticDevice, 0);

	// retrieve information about the current haptic device
	cHapticDeviceInfo hapticDeviceInfo = hapticDevice->getSpecifications();

	// create a tool (cursor) and insert into the world

	tool = new cToolCursor(world);

	world->addChild(tool);

	// connect the haptic device to the virtual tool
	tool->setHapticDevice(hapticDevice);

	// if the haptic device has a gripper, enable it as a user switch
	hapticDevice->setEnableGripperUserSwitch(true);

	// define a radius for the virtual tool (sphere)
	double toolRadius = 0.02;
	tool->setRadius(toolRadius);

	// map the physical workspace of the haptic device to a larger virtual workspace.
	tool->setWorkspaceRadius(0.6);

	// oriente tool with camera
	tool->setLocalRot(camera->getLocalRot());

	// haptic forces are enabled only if small forces are first sent to the device;
	// this mode avoids the force spike that occurs when the application starts when 
	// the tool is located inside an object for instance. 
	tool->setWaitForSmallForce(true);




	// start the haptic tool
	tool->start();

	// read the scale factor between the physical workspace of the haptic
	// device and the virtual workspace defined for the tool
	double workspaceScaleFactor = tool->getWorkspaceScaleFactor();

	// stiffness properties
	maxStiffness = hapticDeviceInfo.m_maxLinearStiffness / workspaceScaleFactor;


	//--------------------------------------------------------------------------
	// CREATE OBJECT
	//--------------------------------------------------------------------------

	// create a volumetric model
	object = new cVoxelObject();



	// add object to world
	world->addChild(object);

	// position object
	//object->setLocalPos(0.0, -0.2, 0.2);
	//object->rotateExtrinsicEulerAnglesDeg(-90, 45, -45, C_EULER_ORDER_XYZ);

	// rotate object


	// set the dimensions by assigning the position of the min and max corners

	object->m_minCorner.set(-0.5, -0.5, -0.5);
	object->m_maxCorner.set(0.5, 0.5, 0.5);

	//object->m_minCorner.set(-1.0, -1.0, -1.0);
	//object->m_maxCorner.set(1.0, 1.0, 1.0);
	object->createAABBCollisionDetector(toolRadius);
	//cCollisionAABBBox object_box = cCollisionAABBBox(cVector3d(-0.5f, -0.5f, -0.5f), cVector3d(0.5f, 0.5f, 0.5f));
	

	// set the texture coordinate at each corner.
	object->m_minTextureCoord.set(0.0, 0.0, 0.0);
	object->m_maxTextureCoord.set(1.0, 1.0, 1.0);

	// set haptic properties
	object->m_material->setStiffness(0.2 * maxStiffness);
	object->m_material->setStaticFriction(0.0);
	object->m_material->setDynamicFriction(0.0);

	// enable materials
	object->setUseMaterial(true);

	// set material color
	object->m_material->setYellowPeachPuff();

	

	

	//--------------------------------------------------------------------------
	// LOAD VOXEL DATA
	//--------------------------------------------------------------------------
	
	// create multi image
	cMultiImagePtr image = cMultiImage::create();


	// For research simple voxel
	custom_image = cMultiImage::create();
	custom_image->allocate(voxResolution, voxResolution, voxResolution, GL_RGBA);

	custom_texture = cTexture3d::create();

	custom_texture->setImage(custom_image);
	object->setTexture(custom_texture);

	buildVoxelCube();
//
//	int filesloaded = image->loadFromFile(RESOURCE_PATH("../resources/volumes/head.raw"));
//	if (filesloaded == 0) {
//#if defined(_MSVC)
//		filesloaded = image->loadFromFile("../../../bin/resources/volumes/head.raw");
//#endif
//	}
//	if (filesloaded == 0) {
//		cout << "Error - Failed to load volume data handXXXX.png." << endl;
//		close();
//		return -1;
//	}


	int filesloaded = image->loadFromFiles(RESOURCE_PATH("../resources/volumes/hand/hand0"), "png", 640);
	if (filesloaded == 0) {
#if defined(_MSVC)
		filesloaded = image->loadFromFiles("../../../bin/resources/volumes/hand/hand0", "png", 640);
#endif
	}
	if (filesloaded == 0) {
		cout << "Error - Failed to load volume data handXXXX.png." << endl;
		close();
		return -1;
	}


//	int filesloaded = image->loadFromFiles(RESOURCE_PATH("../resources/volumes/heart/heart0"), "png", 179);
//	if (filesloaded == 0) {
//#if defined(_MSVC)
//		filesloaded = image->loadFromFiles("../../../bin/resources/volumes/heart/heart0", "png", 179);
//#endif
//	}
//	if (filesloaded == 0) {
//		cout << "Error - Failed to load volume data heartXXXX.png." << endl;
//		close();
//		return -1;
//	}


//	int filesloaded = image->loadFromFiles(RESOURCE_PATH("../resources/volumes/head/cthead-8bit"), "png", 100);
//	if (filesloaded == 0) {
//#if defined(_MSVC)
//		filesloaded = image->loadFromFiles("../../../bin/resources/volumes/head/cthead-8bit", "png", 100);
//#endif
//	}
//	if (filesloaded == 0) {
//		cout << "Error - Failed to load volume data heartXXXX.png." << endl;
//		close();
//		return -1;
//	}

	//--------------------------------------------------------------------------
// LOAD COLORMAPS
//--------------------------------------------------------------------------

	boneLUT = cImage::create();
	bool fileLoaded = boneLUT->loadFromFile(RESOURCE_PATH("../resources/volumes/heart/colormap_bone.png"));
	if (!fileLoaded) {
#if defined(_MSVC)
		fileLoaded = boneLUT->loadFromFile("../../../bin/resources/volumes/heart/colormap_bone.png");
#endif
	}
	if (!fileLoaded)
	{
		cout << "Error - Failed to load colormap." << endl;
		close();
		return -1;
	}

	softLUT = cImage::create();
	fileLoaded = softLUT->loadFromFile(RESOURCE_PATH("../resources/volumes/heart/colormap_soft.png"));
	if (!fileLoaded) {
#if defined(_MSVC)
		fileLoaded = softLUT->loadFromFile("../../../bin/resources/volumes/heart/colormap_soft.png");
#endif
	}
	if (!fileLoaded)
	{
		cout << "Error - Failed to load colormap." << endl;
		close();
		return -1;
	}

	//object->m_colorMap->setImage(softLUT);
	// create texture for storing haptic information

	
	// create texture
	cTexture3dPtr texture = cTexture3d::create();

	// assign volumetric image to texture
	//texture->setImage(image);

	//// assign texture to voxel object
	//object->setTexture(texture);

	//// set isosurface level on object
	//object->setIsosurfaceValue(0.18);

	// set optical density factor
	//object->setOpticalDensity(1.2);
	object->m_material->setStiffness(0.18 * maxStiffness);
	//object->scale(2.0);

	// create texture
	//texture = cTexture3d::create();

	// assign volumetric image to texture
	//texture->setImage(image);

	// assign texture to voxel object
	//object->setTexture(texture);

	// show/hide boundary box
	object->setShowBoundaryBox(true);

	// compute collision detection algorithm
	//object->createAABBCollisionDetector(toolRadius);

	// enable isosurface rendering mode
	//object->setRenderingModeIsosurfaceMaterial();
	//object->setRenderingModeDVRColorMap();
	object->setRenderingModeIsosurfaceColors();
	object->setUseTransparency(true);
	object->setTransparencyLevel(0.5f);
	
	// collision detection 알고리즘 작동하도록 해야하고, 거기서 collision 어디서 났는지 좌표(복셀)를 받아 올 수 있도록 해야함. 
	int v_width = 512;
	int v_height = 512;
	int v_depth = 640;

	cColorb *src = (cColorb*)malloc(sizeof(cColorb)*v_width*v_height*v_depth);
	cColorb *dst = (cColorb*)malloc(sizeof(cColorb)*v_width*v_height*v_depth);
	cColorb* src_d;
	cColorb* dst_d;

	
	// Force Arrow
	arrow = new cMultiMesh();
	bool fileload;
	fileload = arrow->loadFromFile(RESOURCE_PATH("../resources/models/arrow/arrow.obj"));

	if (!fileload)
	{
#if defined(_MSVC)
		fileload = arrow->loadFromFile("../../../bin/resources/models/arrow/arrow.obj");
#endif
	}
	if (!fileload)
	{
		cout << "Error - 3D Model failed to load correctly." << endl;
		close();
		return (-1);
	}


	// disable culling so that faces are rendered on both sides
	arrow->setUseCulling(false);

	cColorf color_arrow;
	color_arrow.setRed();
	arrow->setUseVertexColors(true);
	arrow->setVertexColor(color_arrow);

	// scale model


	// use display list for faster rendering
	arrow->setUseDisplayList(true);

	// position object in scene
	//arrow->rotateExtrinsicEulerAnglesDeg(0, 90, 0, C_EULER_ORDER_XYZ);

	arrow->scaleXYZ(0.2, 0.05, 0.05);

	cTransform toolTrans;
	toolTrans = tool->getGlobalTransform();
	

	
	world->addChild(arrow);



	//--------------------------------------------------------------------------
	// WIDGETS
	//--------------------------------------------------------------------------

	// create a font
	cFont *font = NEW_CFONTCALIBRI20();

	// create a label to display the haptic rate of the simulation
	labelHapticRate = new cLabel(font);
	camera->m_frontLayer->addChild(labelHapticRate);



	// set font color
	labelHapticRate->m_fontColor.setBlack();

	// create a label to display the graphic rate
	labelGraphicRate = new cLabel(font);
	camera->m_frontLayer->addChild(labelGraphicRate);

	labelGraphicRate->m_fontColor.setBlack();

	// create a small message
	labelMessage = new cLabel(font);
	labelMessage->m_fontColor.setBlack();
	labelMessage->setText("press keys [1,2] to toggle between isometric values");
	camera->m_frontLayer->addChild(labelMessage);

	// create a background
	cBackground* background = new cBackground();
	camera->m_backLayer->addChild(background);

	// set background properties
	background->setCornerColors(cColorf(1.0f, 1.0f, 1.0f),
		cColorf(1.0f, 1.0f, 1.0f),
		cColorf(0.7f, 0.7f, 0.7f),
		cColorf(0.7f, 0.7f, 0.7f));

	//if (writeFile.is_open())
	//{
	//	writeFile << 0.9 << " " << 7000.0 << "\n";
	//	writeFile << 100 << "\n";
	//	for (int i = 0; i < 100; i++)
	//	{
	//		writeFile << ((double) (i % 10)) << " " << ((double)(i / 10)) << "\n";
	//	}

	//	writeFile << (1 + 2 * 8 + 1)*9 << "\n";

	//	for (int i = 0; i < 100 - 10; i++)
	//	{
	//		
	//		if (i % 10 == 0)
	//		{
	//			writeFile << i << " " << i + 1 << " " << i + 10 << "\n";

	//		}
	//		else if (i % 10 == 9)
	//		{
	//			writeFile << i << " " << i + 10 << " " << i + 10 - 1 << "\n";
	//		}
	//		else
	//		{
	//			writeFile << i << " " << i + 10 << " " << i + 10 - 1 << "\n";
	//			writeFile << i << " " << i + 1 << " " << i + 10 << "\n";

	//		}

	//	}

	//	writeFile << 10 << "\n";
	//	writeFile << 0 << " " << 3 << "\n";
	//	writeFile << 1 << " " << 3 << "\n";
	//	writeFile << 2 << " " << 3 << "\n";
	//	writeFile << 3 << " " << 3 << "\n";
	//	writeFile << 4 << " " << 3 << "\n";
	//	writeFile << 5 << " " << 3 << "\n";
	//	writeFile << 6 << " " << 3 << "\n";
	//	writeFile << 7 << " " << 3 << "\n";
	//	writeFile << 8 << " " << 3 << "\n";
	//	writeFile << 9 << " " << 3 << "\n";

	//	writeFile << 2 << "\n";
	//	
	//	writeFile << 94 << " " << 0.0 << " " << -3.0 << " " << "\n";
	//	writeFile << 95 << " " << 0.0 << " " << -3.0 << " " << "\n";




	//}

	//--------------------------------------------------------------------------
	// START SIMULATION
	//--------------------------------------------------------------------------

	// create a thread which starts the main haptics rendering loop
	
	tool->initialize();
	prevpos = tool->m_hapticPoint->getGlobalPosProxy();
	prevscale = 1.0;
	arrow->translate(prevpos);
	prev_mat = SKIN;
	cur_mat = SKIN;
	debug_vox_pos = 0;
	arrowPos = cVector3d(0, 0, 0);
	arrowPrevPos = cVector3d(0, 0, 0);
	//arrow->translate(arrowPrevPos);
	//returnForce = tool->getDeviceGlobalForce();
	returnForce = cVector3d(0.0, 0.0, -5.0f);
	zero_force_position = tool->getDeviceGlobalPos();
	cThread* hapticsThread = new cThread();
	hapticsThread->start(updateHaptics, CTHREAD_PRIORITY_HAPTICS);

	// setup callback when application exits
	atexit(close);

	// start the main graphics rendering loop
	glutTimerFunc(50, graphicsTimer, 0);
	glutMainLoop();

	// exit
	return (0);
}

//------------------------------------------------------------------------------

void resizeWindow(int w, int h)
{
	windowW = w;
	windowH = h;
}

//------------------------------------------------------------------------------

void keySelect(unsigned char key, int x, int y)
{
	// option ESC: exit
	if ((key == 27) || (key == 'x'))
	{
		// exit application
		exit(0);
	}

	// option 1: render skin isosurface
	if (key == '1')
	{
		object->setIsosurfaceValue(0.18);
		//object->m_material->setStiffness(0.18 * maxStiffness);
		object->m_material->setYellowPeachPuff();
		cout << "> Isosurface set to " << cStr(object->getIsosurfaceValue(), 3) << "        \r";
	}

	// option 2: render bone isosurface
	if (key == '2')
	{
		object->setIsosurfaceValue(0.32);
		//object->m_material->setStiffness(0.32 * maxStiffness);
		object->m_material->setWhiteIvory();
		cout << "> Isosurface set to " << cStr(object->getIsosurfaceValue(), 3) << "        \r";
	}

	// option q: decrease quality of graphic rendering
	if (key == 'q')
	{
		double value = object->getQuality();
		object->setQuality(value - 0.1);
		cout << "> Quality set to " << cStr(object->getQuality(), 1) << "        \r";
	}

	// option w: increase quality of graphic rendering
	if (key == 'w')
	{
		double value = object->getQuality();
		object->setQuality(value + 0.1);
		cout << "> Quality set to " << cStr(object->getQuality(), 1) << "        \r";
	}

	// option f: toggle fullscreen
	if (key == 'f')
	{
		if (fullscreen)
		{
			windowPosX = glutGet(GLUT_INIT_WINDOW_X);
			windowPosY = glutGet(GLUT_INIT_WINDOW_Y);
			windowW = glutGet(GLUT_INIT_WINDOW_WIDTH);
			windowH = glutGet(GLUT_INIT_WINDOW_HEIGHT);
			glutPositionWindow(windowPosX, windowPosY);
			glutReshapeWindow(windowW, windowH);
			fullscreen = false;
		}
		else
		{
			glutFullScreen();
			fullscreen = true;
		}
	}

	// option m: toggle vertical mirroring
	if (key == 'm')
	{
		mirroredDisplay = !mirroredDisplay;
		camera->setMirrorVertical(mirroredDisplay);
	}
}

//------------------------------------------------------------------------------

void mouseClick(int button, int state, int x, int y)
{
	mouseX = x;
	mouseY = y;
}

//------------------------------------------------------------------------------

void mouseMove(int x, int y)
{
	// compute mouse motion
	int dx = x - mouseX;
	int dy = y - mouseY;
	mouseX = x;
	mouseY = y;

	// compute ne camera angles
	double azimuthDeg = camera->getSphericalAzimuthDeg() + (0.5 * dy);
	double polarDeg = camera->getSphericalPolarDeg() + (-0.5 * dx);

	// assign new angles
	camera->setSphericalAzimuthDeg(azimuthDeg);
	camera->setSphericalPolarDeg(polarDeg);

	// line up tool with camera
	tool->setLocalRot(camera->getLocalRot());
}

//------------------------------------------------------------------------------

void close(void)
{
	// stop the simulation
	simulationRunning = false;

	// wait for graphics and haptics loops to terminate
	while (!simulationFinished) { cSleepMs(100); }

	// close haptic device
	tool->stop();
}

//------------------------------------------------------------------------------

void graphicsTimer(int data)
{
	if (simulationRunning)
	{
		glutPostRedisplay();
	}

	glutTimerFunc(50, graphicsTimer, 0);
}

//------------------------------------------------------------------------------

static bool test = false;

void updateGraphics(void)
{

	if (test) { return; }
	test = true;

	/////////////////////////////////////////////////////////////////////
	// UPDATE WIDGETS
	/////////////////////////////////////////////////////////////////////

	// update haptic rate data
	labelHapticRate->setText(cStr(frequencyCounter.getFrequency(), 0) + " Hz");

	// update position of label
	labelHapticRate->setLocalPos((int)(0.5 * (windowW - labelHapticRate->getWidth())), 15);


	// update graphic rate data
	labelGraphicRate->setText(cStr(frequencyCounter_graphic.getFrequency(), 0) + " Hz");

	// update position of label
	labelGraphicRate->setLocalPos((int)(0.5 * (windowW - labelHapticRate->getWidth())), 40);

	// update position of message label
	labelMessage->setLocalPos((int)(0.5 * (windowW - labelMessage->getWidth())), 65);
	//arrow->translate(tool->getGlobalPos());
	
	
	

	/////////////////////////////////////////////////////////////////////
	// RENDER SCENE
	/////////////////////////////////////////////////////////////////////
	frequencyCounter_graphic.signal(1);
	// update shadow maps (if any)
	world->updateShadowMaps(false, mirroredDisplay);

	// render world
	camera->renderView(windowW, windowH);

	// swap buffers
	glutSwapBuffers();

	// wait until all GL commands are completed
	glFinish();

	// check for any OpenGL errors
	GLenum err = glGetError();
	if (err != GL_NO_ERROR) cout << "Error: " << gluErrorString(err) << endl;

	test = false;

}

//------------------------------------------------------------------------------

enum cMode
{
	IDLE,
	SELECTION
};


void updateHaptics(void)
{
	cMode state = IDLE;
	cGenericObject* selectedObject = NULL;
	cTransform tool_T_object;

	// simulation in now running
	simulationRunning = true;
	simulationFinished = false;

	// haptic force activation
	bool flagStart = true;
	int counter = 0;

	bool flag_diff = false;

	nodesX.resize(100);
	nodesX.setZero();
	nodesY.resize(100);
	nodesY.setZero();
	loads.resize(2 * 100);
	loads.setZero();

	// main haptic simulation loop
	while (simulationRunning)
	{
		mutexVoxel.acquire();
		/////////////////////////////////////////////////////////////////////
		// READ HAPTIC DEVICE
		/////////////////////////////////////////////////////////////////////
		start = time(NULL);
		 // read position 
		cVector3d position;
		hapticDevice->getPosition(position);
		
		// read orientation 
		cMatrix3d rotation;
		hapticDevice->getRotation(rotation);

		// read gripper position
		double gripperAngle;
		hapticDevice->getGripperAngleRad(gripperAngle);

		// read linear velocity 
		cVector3d linearVelocity;
		hapticDevice->getLinearVelocity(linearVelocity);

		// read angular velocity
		cVector3d angularVelocity;
		hapticDevice->getAngularVelocity(angularVelocity);

		// read gripper angular velocity
		double gripperAngularVelocity;
		hapticDevice->getGripperAngularVelocity(gripperAngularVelocity);


		

		/////////////////////////////////////////////////////////////////////////
		// HAPTIC RENDERING
		/////////////////////////////////////////////////////////////////////////

		// update frequency counter
		frequencyCounter.signal(1);

		// compute global reference frames for each object
		world->computeGlobalPositions(true);


		
		// compute interaction forces
		if (flag == 0)
		{

			tool->updateFromDevice();
			tool->computeInteractionForces();

		}
		else
		{

			tool->initialize();
			tool->updateFromDevice();

			
		}

		if (tool->getDeviceLocalForce().z() >= 3)
			flag = 1;
		
		
		cVector3d curpos;
		curpos = tool->m_hapticPoint->getGlobalPosProxy();

		cCollisionSettings collisionSettings;
		collisionSettings.m_checkForNearestCollisionOnly = false;
		collisionSettings.m_returnMinimalCollisionData = false;
		collisionSettings.m_checkVisibleObjects = false;
		collisionSettings.m_checkHapticObjects = true;
		collisionSettings.m_ignoreShapes = true;
		collisionSettings.m_adjustObjectMotion = false;
		collisionSettings.m_collisionRadius = 0.01;

		// setup recorder
		cCollisionRecorder collisionRecorder;
		collisionRecorder.clear();
		
		cVector3d nextProxyOffset(0.0, 0.0, 0.0);

		// check if any moving objects have hit the proxy
		

		bool hit = object->computeCollisionDetection(tool->getDeviceGlobalPos()+cVector3d(0,0,0.02), tool->getDeviceGlobalPos() - cVector3d(0, 0, 0.02), collisionRecorder, collisionSettings);
		contact = &collisionRecorder.m_collisions[0];
		arrow->translate(-prevpos);
		arrow->translate(curpos);
		//arrow->translate(-curpos);
		//arrow->translate(arrowPos);
		//arrow->translate(-arrowPrevPos);
		//arrow->translate(arrowPos);
	
		arrow->scale(1.0 / prevscale);
		
		arrowPrevPos = arrowPos;
		

		cVector3d directionVec = curpos - prevpos;
		directionVec.normalize();
		

		cVector3d force = returnForce;



		float force_x = force.x();
		float force_z = force.z();

		float angle = atan(force_z / force_x) * 180.0f / M_PI;
		
		//cout << angle << endl;
		arrow->rotateExtrinsicEulerAnglesDeg(0, -angle, 0, C_EULER_ORDER_XYZ);
		double scalefactor = 1.0;
		if (force.z() == 0)
		{
			scalefactor = 1.0;
		}
		else
		{
			scalefactor = (double) force_z/2.0;
		}
		arrow->scale(scalefactor);
		prevscale = scalefactor;

		//cout << tool->getDeviceGlobalPos() << endl;

		float poissonRatio, youngModulus;

		mutexVoxel.acquire();
		
		int nodescnt = 100;
		//nodescnt = voxResolution/5 * voxResolution/10;
		int num_vox_x = 10;
		//num_vox_x = voxResolution/10;
		
		
		if (hit && flag == 1)
		{
			//cout << force.z() << endl;
			hit_flag = 1;
			cVector3d VoxPos = VoxIdx_to_GPos();
			arrowPos = VoxIdx_to_GPos2(127, 127, 127);
			cColorb voxColor(0, 0, 0, 000);
			cColorb arrowColor(0, 0, 0, 0);


			elements.clear();
			constraints.clear();
			constraint_nodes.clear();
			object->m_texture->m_image->getVoxelColor(contact->m_voxelIndexX, contact->m_voxelIndexY, contact->m_voxelIndexZ, voxColor);
			//object->m_texture->m_image->getVoxelColor(127, 127, 127, voxColor);

			cout << cColorBtoF(voxColor.getR()) << endl;
			

			
			Material mat = materialTransferFunc(cColorBtoF(voxColor.getR()));
			cur_mat = mat;
			if (cur_mat != prev_mat)
				flag_diff = true;
			else
				flag_diff = false;

			if (mat == SKIN)
			{
				poissonRatio = 0.3f;
				//youngModulus = 0.1f;
				youngModulus = 100.0f;
			}

			else if (mat == BRAIN)
			{
				poissonRatio = 0.3f;
				youngModulus = 1000.0f;
			}

			else if (mat == BONE)
			{
				poissonRatio = 0.9f;
				youngModulus = 7000.0f;

			}

			Eigen::Matrix3f D;
			D <<
				1.0f, poissonRatio, 0.0f,
				poissonRatio, 1.0, 0.0f,
				0.0f, 0.0f, (1.0f - poissonRatio) / 2.0f;

			D *= youngModulus / (1.0f - pow(poissonRatio, 2.0f));
			//nodesX.resize(nodescnt);
			//nodesX.setZero();
			//nodesY.resize(nodescnt);
			//nodesY.setZero();
			//loads.resize(2 * nodescnt);
			//loads.setZero();

			
			for (int i = 0; i < nodescnt; i++)
			{
				nodesX[i] = (float) (i % num_vox_x);
				nodesY[i] = (float)(i / num_vox_x);
			}




			for (int i = 0; i < nodescnt - num_vox_x; i++)
			{
				
				if (i % num_vox_x == 0)
				{
					Element element;
					element.nodesIds[0] = i;
					element.nodesIds[1] = i + 1;
					element.nodesIds[2] = i + num_vox_x;
					elements.push_back(element);
				}
				else if (i % num_vox_x == num_vox_x - 1)
				{
					Element element;
					element.nodesIds[0] = i;
					element.nodesIds[1] = i + num_vox_x;
					element.nodesIds[2] = i + num_vox_x-1;
					elements.push_back(element);
				}
				else
				{
					Element element;
					element.nodesIds[0] = i;
					element.nodesIds[1] = i + num_vox_x;
					element.nodesIds[2] = i + num_vox_x - 1;
					elements.push_back(element);
					Element element2;
					element2.nodesIds[0] = i;
					element2.nodesIds[1] = i + 1;
					element2.nodesIds[2] = i + num_vox_x;
					elements.push_back(element);

				}
				
				
			}


			int constraint_cnt = 0;

			for (int i = 0; i < nodescnt; i++)
			{
				if (i / num_vox_x == 0)
				{
					Constraint constraint;
					int type;
					constraint.node = i;
					type = 3;
					constraint.type = static_cast<Constraint::Type>(type);
					constraints.push_back(constraint);
					constraint_nodes.push_back(i);
					constraint_cnt++;
				}
				else
				{
					Material constraint_mat;
					int pos_index = i % num_vox_x;
					int pos_index2 = i / num_vox_x;
					if (pos_index == num_vox_x/2 || pos_index == num_vox_x/2 - 1)
					{
						object->m_texture->m_image->getVoxelColor(contact->m_voxelIndexX, contact->m_voxelIndexY, contact->m_voxelIndexZ - (nodescnt / num_vox_x) + pos_index2, voxColor);
						//object->m_texture->m_image->getVoxelColor(127, 127, 127 - (nodescnt / num_vox_x) + pos_index2, voxColor);

					}
					else if (pos_index < (num_vox_x/2 - 1))
					{
						//object->m_texture->m_image->getVoxelColor(127 - (num_vox_x - 1 - pos_index), 127, 127 - (nodescnt / num_vox_x) + pos_index2, voxColor);

						object->m_texture->m_image->getVoxelColor(contact->m_voxelIndexX -(num_vox_x-1 - pos_index), contact->m_voxelIndexY, contact->m_voxelIndexZ - (nodescnt / num_vox_x) + pos_index2, voxColor);
					}
					else if (pos_index > (num_vox_x/2))
					{
						object->m_texture->m_image->getVoxelColor(contact->m_voxelIndexX - (num_vox_x - pos_index), contact->m_voxelIndexY, contact->m_voxelIndexZ - (nodescnt / num_vox_x) + pos_index2, voxColor);

						//object->m_texture->m_image->getVoxelColor(127 - (num_vox_x - pos_index), 127, 127 - (nodescnt / num_vox_x) + pos_index2, voxColor);
					}
					
					constraint_mat = materialTransferFunc(cColorBtoF(voxColor.getR()));
					if (constraint_mat != mat)
					{
						
						Constraint constraint;
						int type;
						constraint.node = i;
						type = 3;
						constraint.type = static_cast<Constraint::Type>(type);
						constraints.push_back(constraint);
						constraint_nodes.push_back(i);
						constraint_cnt++;
					}
				}
	
			}
			//cout << constraint_cnt << endl;
			if (mat == SKIN)
			{
				loads[2 * (nodescnt - num_vox_x / 2) + 0] = 0;
				loads[2 * (nodescnt - num_vox_x / 2) + 1] = -3.0f;

				loads[2 * (nodescnt - num_vox_x / 2 + 1) + 0] = 0;
				loads[2 * (nodescnt - num_vox_x / 2 + 1) + 1] = -3.0f;
			}
			else
			{
				loads[2 * (nodescnt - num_vox_x / 2) + 0] = 0;
				loads[2 * (nodescnt - num_vox_x / 2) + 1] = -3.0f;

				loads[2 * (nodescnt - num_vox_x / 2 + 1) + 0] = 0;
				loads[2 * (nodescnt - num_vox_x / 2 + 1) + 1] = -3.0f;
			}




			std::vector<Eigen::Triplet<float> > triplets;
			for (std::vector<Element>::iterator it = elements.begin(); it != elements.end(); ++it)
			{
				it->CalculateStiffnessMatrix(D, triplets);
			}

			
			Eigen::SparseMatrix<float> globalK(2 * nodescnt, 2 * nodescnt);
			globalK.setFromTriplets(triplets.begin(), triplets.end());

			ApplyConstraints(globalK, constraints);

			Eigen::SimplicialLDLT<Eigen::SparseMatrix<float> > solver(globalK);

			Eigen::VectorXf displacements = solver.solve(loads);

			cVector3d goalPos = cVector3d(0, 0, VoxPos.z() + displacements(nodescnt-1)/((float)voxResolution));
			cVector3d toolPos = tool->getDeviceGlobalPos();


			float sigma_mises;
			float sum = 0;
			int cnt = 0;
			int cnt2 = 0;
			for (std::vector<Element>::iterator it = elements.begin(); it != elements.end(); ++it)
			{
				Eigen::Matrix<float, 6, 1> delta;
				delta << displacements.segment<2>(2 * it->nodesIds[0]),
					displacements.segment<2>(2 * it->nodesIds[1]),
					displacements.segment<2>(2 * it->nodesIds[2]);

				Eigen::Vector3f sigma = D * it->B * delta;
				if (cnt == (2 + (num_vox_x - 2) * 2)*(nodescnt / num_vox_x - 1) - (num_vox_x / 2 - 1) * 2)
				{
					sigma_mises = sqrt(sigma[0] * sigma[0] - sigma[0] * sigma[1] + sigma[1] * sigma[1] + 3.0f * sigma[2] * sigma[2]);
				}
				int elem_node = it->nodesIds[0];
				bool check_flag = true;
				if (std::find(constraint_nodes.begin(), constraint_nodes.end(), elem_node) != constraint_nodes.end()) {
					/* v contains x */
					//cout << "here" << endl;
					//cout << elem_node << endl;
					check_flag = false;
				}
				else {
					/* v does not contain x */
					check_flag = true;
				}
				if ((elem_node % num_vox_x == num_vox_x/2) && (check_flag))
				{
					//cout << elem_node << endl;
					cnt2++;
					sum += sqrt(sigma[0] * sigma[0] - sigma[0] * sigma[1] + sigma[1] * sigma[1] + 3.0f * sigma[2] * sigma[2]);
				}
				
				//cout << sqrt(sigma[0] * sigma[0] - sigma[0] * sigma[1] + sigma[1] * sigma[1] + 3.0f * sigma[2] * sigma[2]) << endl;
				cnt++;
				
			}
			sum = sum / cnt2;
			sigma_mises = sum;
			//cout << force.z() << endl;
			//cout << sum << endl;
			//cout << "Pos: " << contact->m_voxelIndexZ << endl;
			//cout << "sigma: " << sigma_mises << endl;
		
			//cout << "Force: " << force.z() << endl;
			//cout << "Displacemet: " << displacements(2 * nodescnt - num_vox_x) << endl;
			if (writeFile.is_open())
			{
				writeFile << contact->m_voxelIndexZ << "," << force.z() << "\n";

			}
			
			
			//cout << cnt << endl;
			//cout << tool->getDeviceGlobalLinVel() << endl;
			//cout << "displament: " << displacements(nodescnt-1)/255.0f << endl;
			//cout << "sigma: " << sigma_mises << endl;
			float movement = (zero_force_position - tool->getDeviceGlobalPos()).length();

			if (movement > 0.f)
			{
				if (goalPos.z() > toolPos.z())
				{

					float godOForce = (1.0f+(displacements(2 * nodescnt - num_vox_x+1)*5))*sigma_mises;
					float godOForceX = (0.0f + (displacements(2 * nodescnt - num_vox_x+1)*5))*sigma_mises;
					if (flag_diff || cnt2 == 0)
					{
						//cout << "asdf" << endl;
						if (mat == SKIN)
						{
							godOForce = 2.0f;
						}
						else
						{
							godOForce = 2.0f;
						}
					}
					//cout << godOForce << endl;
					//cout << "sigma: " << sigma_mises << endl;
					//cout << "Displacemet: " << displacements(2 * nodescnt - num_vox_x+1) << endl;
					tool->setDeviceGlobalForce(cVector3d(0, 0, godOForce));
					tool->setDeviceGlobalPos(goalPos);
					returnForce = cVector3d(0, 0, godOForce);
					
				}
				else if (VoxPos.z() > toolPos.z())
				{

					float godOForce = (1.0f + (displacements(2 * nodescnt - num_vox_x+1)*5))*sigma_mises;
					float godOForceX = (0.0f + (displacements(2 * nodescnt - num_vox_x + 1)*5))*sigma_mises;
					if (flag_diff || cnt2 == 0)
					{
						if (mat == SKIN)
						{
							godOForce = 2.0f;
						}
						else
						{
							godOForce = 2.0f;
						}
					}
					//cout << godOForce << endl;
					//cout << "sigma: " << sigma_mises << endl;
					//cout << "Displacemet: " << displacements(2 * nodescnt - num_vox_x+1) << endl;
					tool->setDeviceGlobalForce(cVector3d(0, 0, godOForce));
					returnForce = cVector3d(0, 0, godOForce);
					//tool->setDeviceGlobalForce(returnForce);
				}
				else
				{
					//tool->setDeviceGlobalForce(cVector3d(0, 0, 0));
					//returnForce = cVector3d(0, 0, 0);
					tool->setDeviceGlobalForce(returnForce);
				}
				zero_force_position = tool->getDeviceGlobalPos();
			}

			else
			{
				tool->setDeviceGlobalForce(cVector3d(0, 0, 0));
				returnForce = cVector3d(0, 0, 0);
			}
			prev_mat = mat;
			double result = (double)(time(NULL) - start);
			//cout << result << endl;

			//if ((clock() - start) > 70)
			//{
			//	//cout << "arrowPos: " << sigma_mises << endl;
			//	//cout << "toolPos: " << tool->m_hapticPoint->getGlobalPosProxy().y() << endl;
			//
			//	cout << force.z() << endl;
			//	debug_vox_pos += 1;
			//	start = clock();
			//}
			
			//tool->setDeviceGlobalForce(cVector3d(0, 0, 5.0f));
			//cout << materialTransferFunc(cColorBtoF(voxColor.getR())) << endl;;
			
			//std::cout << cColorBtoF(voxColor.getR()) << " " << cColorBtoF(voxColor.getG()) << " " << cColorBtoF(voxColor.getB()) << std::endl;
		}
		else
		{
			
		//if (flag == 1)
		//	{
		//		tool->setDeviceGlobalForce(cVector3d(0, 0, 0));
		//		returnForce = cVector3d(0, 0, 0);
		//	}
			flag = 0;
			prev_mat = SKIN;
			//flag = 0;
			//if (hit_flag == 1)
			//{
			//	flag = 0;
			//	hit_flag = 0;
			//}
			
		}

		//cout << tool->getDeviceGlobalForce() << endl;
		//tool->m_hapticPoint->m_sphereProxy->setLocalPos(tool->getDeviceGlobalPos());


		/////////////////////////////////////////////////////////////////////////
		// FINALIZE
		/////////////////////////////////////////////////////////////////////////


		// send forces to haptic device
		//tool->initialize();
		prevpos = curpos;
		tool->setDeviceGlobalPos(cVector3d(0.0, 0.0, 0.0));
		//tool->setDeviceGlobalForce(cVector3d(0.0, 3.0, 0.0));
		tool->applyToDevice();
		mutexVoxel.release();
	}

	// exit haptics thread
	simulationFinished = true;
}

cVector3d cVec3Clamp(cVector3d &a_value, cVector3d &a_low, cVector3d &a_high)
{
	float a_x = a_value.x();
	float low_x = a_low.x();
	float high_x = a_high.x();
	float a_y = a_value.y();
	float low_y = a_low.y();
	float high_y = a_high.y();
	float a_z = a_value.z();
	float low_z = a_low.z();
	float high_z = a_high.z();

	float x_val = (a_x < low_x ? low_x : a_x > high_x ? high_x : a_x);
	float y_val = (a_y < low_y ? low_y : a_y > high_y ? high_y : a_y);
	float z_val = (a_z < low_z ? low_z : a_z > high_z ? high_z : a_z);

	cVector3d clamped = cVector3d(x_val, y_val, z_val);


	return clamped;
}
//Collision Check
bool CollisionCheck(cVoxelObject* voxObject, cToolCursor* tool_cursor)
{

	//Get center point tool
	cVector3d center = tool_cursor->getDeviceGlobalPos();

	// Calculate AABB infor (center, half-extents)
	cVector3d aabb_center = voxObject->getBoundaryCenter();

	// Get difference vector between both centers
	cVector3d difference = center - aabb_center;
	
	cVector3d clamped = cVec3Clamp(difference, voxObject->getBoundaryMin(), voxObject->getBoundaryMax());

	cVector3d closest = aabb_center + clamped;


	difference = closest - center;

	return  difference.length() < tool_cursor->m_hapticPoint->getRadiusContact();
}


Material materialTransferFunc(float intensity)
{
	if (intensity < 0.18)
	{
		return BRAIN;
	}
	else if (intensity < 0.32)
	{
		return SKIN;
	}
	else
	{
		return BONE;
	}
}


float defineProperty(cCollisionEvent* contact2, cVoxelObject* object2)
{
	cColorb voxColor(0x00, 0x00, 0x00, 0x00);
	//object->m_texture->m_image->getVoxelColor(contact->m_voxelIndexX, contact->m_voxelIndexY, contact->m_voxelIndexZ, voxColor);
	float matproperty = 0;
	int cnt = 0;
	for (int x = -11; x < 12; x++)
	{
		for (int y = -11; y < 12; y++)
		{
			for (int z = -11; z < 12; z++)
			{
				object->m_texture->m_image->getVoxelColor(contact->m_voxelIndexX+x, contact->m_voxelIndexY+y, contact->m_voxelIndexZ+z, voxColor);
				matproperty += cColorBtoF(voxColor.getR());
				cnt += 1;
			}
		}
		
	}
	return matproperty / cnt;
}

float get_density(int x, int y, int z)
{
	cColorb voxColor(0x00, 0x00, 0x00, 0x00);
	float res;
	object->m_texture->m_image->getVoxelColor(x, y, z, voxColor);
	res = cColorBtoF(voxColor.getR());
	return res;
}


cVector3d get_gradient(int x, int y, int z)
{
	cVector3d res;
	float fx1 = get_density(x + 1, y, z);
	float fx2 = get_density(x - 1, y, z);
	float res_x = (fx1 - fx2) / 2.0f;

	float fy1 = get_density(x, y+1, z);
	float fy2 = get_density(x, y-1, z);
	float res_y = (fy1 - fy2) / 2.0f;

	float fz1 = get_density(x, y, z+1);
	float fz2 = get_density(x, y, z-1);
	float res_z = (fz1 - fz2) / 2.0f;

	res = cVector3d(res_x, res_y, res_z);

	return res;
	
}


void buildVoxelCube()
{
	mutexVoxel.acquire();

	// fill all voxels
	for (float z = 0; z < voxResolution; z++)
	{
		for (float y = 0; y < voxResolution; y++)
		{
			for (float x = 0; x < voxResolution; x++)
			{
				double r, g, b, a;

				if (z < 50.0f)
				{
					r = (double)50 / (double)255;
					g = (double)50 / (double)255;
					b = (double)50 / (double)255;
					a = 50.0/255.0;
					//r = (double)150 / (double)255;
					//g = (double)150 / (double)255;
					//b = (double)150 / (double)255;
				}
				else if (z < 150.0f)
				{
					//r = (double)150 / (double)255;
					//g = (double)150 / (double)255;
					//b = (double)150 / (double)255;
					a = 220.0 / 255.0;
					r = (double)150 / (double)255;
					g = (double)150 / (double)255;
					b = (double)150 / (double)255;
				}

				else
				{
					if (z < 50.0f / pow(255.0f, 2)*pow((y + 255.0f / 2.0f), 2)+149.0f)
					{
						//cout << z << endl;
						r = (double)150 / (double)255;
						g = (double)150 / (double)255;
						b = (double)150 / (double)255;
						a = 220.0 / 255.0;
					}
					else
					{
						//cout << z << endl;
						r = (double)50 / (double)255;
						g = (double)50 / (double)255;
						b = (double)50 / (double)255;
						a = 50.0 / 255.0;
					}
					//r = (double)150 / (double)255;
					//g = (double)150 / (double)255;
					//b = (double)150 / (double)255;
					
				}
				//r = (double)150 / (double)255;
				//g = (double)150 / (double)255;
				//b = (double)150 / (double)255;

				cColorb color;
				color.setf(r, g, b, 1.0f);
				custom_image->setVoxelColor(x, y, z, color);

			}
		}
	}

	custom_texture->markForUpdate();

	mutexVoxel.release();
}

cVector3d VoxIdx_to_GPos()
{
	float x = (float) contact->m_voxelIndexX;
	float y = (float) contact->m_voxelIndexY;
	float z = (float) contact->m_voxelIndexZ;

	cVector3d gpos = cVector3d(x / ((float)voxResolution) - 0.5f, y / ((float)voxResolution) - 0.5f, z / ((float)179) - 0.5f);
	return gpos;
}


cVector3d VoxIdx_to_GPos2(int voxX, int voxY, int voxZ)
{
	float x = (float)voxX;
	float y = (float)voxY;
	float z = (float)voxZ;

	cVector3d gpos = cVector3d(x / ((float)voxResolution) - 0.5f, y / ((float)voxResolution) - 0.5f, z / ((float)179) - 0.5f);
	return gpos;
}


//cVector3d get_gradient(int x, int y, int z)
//{
//	cVector3d res;
//	// x
//	int xi = (int)(x + 0.5f);
//	float xf = (float)x + 0.5f - xi;
//	float xd0 = get_density(xi - 1, y, z);
//	float xd1 = get_density(xi, y, z);
//	float xd2 = get_density(xi + 1, y, z);
//	float res_x = (xd1 - xd0) * (1.0f - xf) + (xd2 - xd1) * xf;
//	
//	int yi = (int)(y + 0.5f);
//	float yf = (float)y + 0.5f - yi;
//	float yd0 = get_density(x, yi - 1, z);
//	float yd1 = get_density(x, yi, z);
//	float yd2 = get_density(x, yi + 1, z);
//	float res_y = (yd1 - yd0) * (1.0f - yf) + (yd2 - yd1) * yf;
//
//	int zi = (int)(z + 0.5f);
//	float zf = (float)y + 0.5f - zi;
//	float zd0 = get_density(x, y, zi-1);
//	float zd1 = get_density(x, y, z);
//	float zd2 = get_density(x, y, zi+1);
//	float res_z = (zd1 - zd0) * (1.0f - zf) + (zd2 - zd1) * zf;
//
//	res = cVector3d(res_x, res_y, res_z);
//
//	return res;
//	
//}
//std::array<float, 3> get_gradient(float x, float y, float z) {
//	std::array<float, 3> res;
//	// x
//	int xi = (int)(x + 0.5f);
//	float xf = x + 0.5f - xi;
//	float xd0 = get_density(xi - 1, (int)y, (int)z);
//	float xd1 = get_density(xi, (int)y, (int)z);
//	float xd2 = get_density(xi + 1, (int)y, (int)z);
//	res[0] = (xd1 - xd0) * (1.0f - xf) + (xd2 - xd1) * xf; // lerp
//	// y
//	int yi = (int)(y + 0.5f);
//	float yf = y + 0.5f - yi;
//	float yd0 = get_density((int)x, yi - 1, (int)z);
//	float yd1 = get_density((int)x, yi, (int)z);
//	float yd2 = get_density((int)x, yi + 1, (int)z);
//	res[1] = (yd1 - yd0) * (1.0f - yf) + (yd2 - yd1) * yf; // lerp
//	// z
//	int zi = (int)(z + 0.5f);
//	float zf = z + 0.5f - zi;
//	float zd0 = get_density((int)x, (int)y, zi - 1);
//	float zd1 = get_density((int)x, (int)y, zi);
//	float zd2 = get_density((int)x, (int)y, zi + 1);
//	res[2] = (zd1 - zd0) * (1.0f - zf) + (zd2 - zd1) * zf; // lerp
//	return res;
//}
//------------------------------------------------------------------------------
