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
    \author    Francois Conti
    \version   3.1.1 $Rev: 1925 $
*/
//==============================================================================

//------------------------------------------------------------------------------
#include "chai3d.h"
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


//------------------------------------------------------------------------------
// DECLARED VARIABLES
//------------------------------------------------------------------------------

// a world that contains all objects of the virtual environment
cWorld* world;

// a camera to render the world in the window display
cCamera* camera;

// a light source to illuminate the objects in the world
cDirectionalLight *light;

// a virtual object
cMultiMesh* object;

// a haptic device handler
cHapticDeviceHandler* handler;

// a pointer to the current haptic device
cGenericHapticDevicePtr hapticDevice;

// a virtual tool representing the haptic device in the scene
cToolCursor* tool;

// a label to display the rate [Hz] at which the simulation is running
cLabel* labelHapticRate;

// indicates if the haptic simulation currently running
bool simulationRunning = false;

// indicates if the haptic simulation has terminated
bool simulationFinished = true;

// frequency counter to measure the simulation haptic rate
cFrequencyCounter frequencyCounter;

// display options
bool showEdges = true;
bool showTriangles = true;
bool showNormals = false;

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

// root resource path
string resourceRoot;

// display level for collision tree
int collisionTreeDisplayLevel = 0;


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

// callback when a key is pressed
void keySelect(unsigned char key, int x, int y);

// callback to handle mouse click
void mouseClick(int button, int state, int x, int y);

// callback to handle mouse motion when button is pressed
void mouseMove(int x, int y);

// callback to render graphic scene
void updateGraphics(void);

// callback of GLUT timer
void graphicsTimer(int data);

// function that closes the application
void close(void);

// main haptics simulation loop
void updateHaptics(void);

// Collision check
bool CollisionCheck(cGenericObject* voxObject, cToolCursor* tool_cursor);
cVector3d cVec3Clamp(cVector3d &a_value, cVector3d &a_low, cVector3d &a_high);

//==============================================================================
/*
    DEMO:   21-object.cpp

    This demonstration loads a 3D mesh file by using the file loader
    functionality of the \ref cMultiMesh class. A finger-proxy algorithm is
    used to render the forces. Take a look at this example to understand the
    different functionalities offered by the tool force renderer.

    In the main haptics loop function  "updateHaptics()" , the position
    of the haptic device is retrieved at each simulation iteration.
    The interaction forces are then computed and sent to the device.
    Finally, a simple dynamics model is used to simulate the behavior
    of the object.
*/
//==============================================================================

int main(int argc, char* argv[])
{
    //--------------------------------------------------------------------------
    // INITIALIZATION
    //--------------------------------------------------------------------------

    cout << endl;
    cout << "-----------------------------------" << endl;
    cout << "CHAI3D" << endl;
    cout << "Demo: 21-object" << endl;
    cout << "Copyright 2003-2016" << endl;
    cout << "-----------------------------------" << endl << endl << endl;
    cout << "Keyboard Options:" << endl << endl;
    cout << "[1] - Texture   (ON/OFF)" << endl;
    cout << "[2] - Wireframe (ON/OFF)" << endl;
    cout << "[3] - Collision tree (ON/OFF)" << endl;
    cout << "[+] - Increase collision tree display depth" << endl;
    cout << "[-] - Decrease collision tree display depth" << endl;
    cout << "[s] - Save screenshot to file" << endl;
    cout << "[e] - Enable/Disable display of edges" << endl;
    cout << "[t] - Enable/Disable display of triangles" << endl;
    cout << "[n] - Enable/Disable display of normals" << endl;
    cout << "[f] - Enable/Disable full screen mode" << endl;
    cout << "[m] - Enable/Disable vertical mirroring" << endl;
    cout << "[x] - Exit application" << endl;
    cout << endl << endl;

    // parse first arg to try and locate resources
    resourceRoot = string(argv[0]).substr(0,string(argv[0]).find_last_of("/\\")+1);


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
    camera->setSphericalReferences(cVector3d(0,0,0),    // origin
                                   cVector3d(0,0,1),    // zenith direction
                                   cVector3d(1,0,0));   // azimuth direction

    camera->setSphericalDeg(1.0,    // spherical coordinate radius
                            35,     // spherical coordinate azimuth angle
                            20);    // spherical coordinate polar angle

    // set the near and far clipping planes of the camera
    // anything in front or behind these clipping planes will not be rendered
    camera->setClippingPlanes(0.01, 100);

    // set stereo mode
    camera->setStereoMode(stereoMode);

    // set stereo eye separation and focal length (applies only if stereo is enabled)
    camera->setStereoEyeSeparation(0.03);
    camera->setStereoFocalLength(1.5);

    // set vertical mirrored display mode
    camera->setMirrorVertical(mirroredDisplay);

    // enable multi-pass rendering to handle transparent objects
    camera->setUseMultipassTransparency(true);

    // create a light source
    light = new cDirectionalLight(world);

    // attach light to camera
    camera->addChild(light);

    // enable light source
    light->setEnabled(true);

    // define the direction of the light beam
    light->setDir(-3.0,-0.5, 0.0);

    // set lighting conditions
    light->m_ambient.set(0.4f, 0.4f, 0.4f);
    light->m_diffuse.set(0.8f, 0.8f, 0.8f);
    light->m_specular.set(1.0f, 1.0f, 1.0f);


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

    // define the radius of the tool (sphere)
    double toolRadius = 0.01;

    // define a radius for the tool
    tool->setRadius(toolRadius);

    // hide the device sphere. only show proxy.
    tool->setShowContactPoints(true, false);

    // create a white cursor
    tool->m_hapticPoint->m_sphereProxy->m_material->setWhite();

    // map the physical workspace of the haptic device to a larger virtual workspace.
    tool->setWorkspaceRadius(1.0);

    // oriente tool with camera
    tool->setLocalRot(camera->getLocalRot());

    // haptic forces are enabled only if small forces are first sent to the device;
    // this mode avoids the force spike that occurs when the application starts when 
    // the tool is located inside an object for instance. 
    tool->setWaitForSmallForce(true);

    // start the haptic tool
    tool->start();


    //--------------------------------------------------------------------------
    // CREATE OBJECT
    //--------------------------------------------------------------------------

    // read the scale factor between the physical workspace of the haptic
    // device and the virtual workspace defined for the tool
    double workspaceScaleFactor = tool->getWorkspaceScaleFactor();

    // stiffness properties
    double maxStiffness	= hapticDeviceInfo.m_maxLinearStiffness / workspaceScaleFactor;

    // create a virtual mesh
    object = new cMultiMesh();

    // add object to world
    world->addChild(object);

    // load an object file
    bool fileload;
    fileload = object->loadFromFile(RESOURCE_PATH("../resources/models/leica/leica.3ds"));
    if (!fileload)
    {
        #if defined(_MSVC)
        fileload = object->loadFromFile("../../../bin/resources/models/leica/leica.3ds");
        #endif
    }
    if (!fileload)
    {
        cout << "Error - 3D Model failed to load correctly" << endl;
        close();
        return (-1);
    }

    // disable culling so that faces are rendered on both sides
    object->setUseCulling(false);

    // get dimensions of object
    object->computeBoundaryBox(true);
    double size = cSub(object->getBoundaryMax(), object->getBoundaryMin()).length();

    // resize object to screen
    if (size > 0.001)
    {
        object->scale(1.0 / size);
    }

     // compute a boundary box
    object->computeBoundaryBox(true);

    // show/hide boundary box
    object->setShowBoundaryBox(false);

    // compute collision detection algorithm
    object->createAABBCollisionDetector(toolRadius);

    // define a default stiffness for the object
    object->setStiffness(0.2 * maxStiffness, true);

    // define some haptic friction properties
    object->setFriction(0.1, 0.2, true);

    // enable display list for faster graphic rendering
    object->setUseDisplayList(true);


	


    // center object in scene
    object->setLocalPos(-1.0 * object->getBoundaryCenter());

	

    // rotate object in scene
    object->rotateExtrinsicEulerAnglesDeg(0, 0, 90, C_EULER_ORDER_XYZ);

    // compute all edges of object for which adjacent triangles have more than 40 degree angle
    object->computeAllEdges(40);

    // set line width of edges and color
    cColorf colorEdges;
    colorEdges.setBlack();
    object->setEdgeProperties(1, colorEdges);

    // set normal properties for display
    cColorf colorNormals;
    colorNormals.setOrangeTomato();
    object->setNormalsProperties(0.01, colorNormals);

    // display options
    object->setShowTriangles(showTriangles);
    object->setShowEdges(showEdges);
    object->setShowNormals(showNormals);


	


    //--------------------------------------------------------------------------
    // WIDGETS
    //--------------------------------------------------------------------------

    // create a font
    cFont *font = NEW_CFONTCALIBRI20();
    
    // create a label to display the haptic rate of the simulation
    labelHapticRate = new cLabel(font);
    labelHapticRate->m_fontColor.setBlack();
    camera->m_frontLayer->addChild(labelHapticRate);

    // create a background
    cBackground* background = new cBackground();
    camera->m_backLayer->addChild(background);

    // set background properties
    background->setCornerColors(cColorf(0.95f, 0.95f, 0.95f),
                                cColorf(0.95f, 0.95f, 0.95f),
                                cColorf(0.80f, 0.80f, 0.80f),
                                cColorf(0.80f, 0.80f, 0.80f));


    //--------------------------------------------------------------------------
    // START SIMULATION
    //--------------------------------------------------------------------------

    // create a thread which starts the main haptics rendering loop
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

    // option 1: show/hide texture
    if (key == '1')
    {
        bool useTexture = object->getUseTexture();
        object->setUseTexture(!useTexture);
    }

    // option 2: enable/disable wire mode
    if (key == '2')
    {
        bool useWireMode = object->getWireMode();
        object->setWireMode(!useWireMode, true);
    }

    // option 3: show/hide collision detection tree
    if (key == '3')
    {
        cColorf color = cColorf(1.0, 0.0, 0.0);
        object->setCollisionDetectorProperties(collisionTreeDisplayLevel, color, true);
        bool show = object->getShowCollisionDetector();
        object->setShowCollisionDetector(!show, true);
    }

    // option -:
    if (key == '-')
    {
        collisionTreeDisplayLevel--;
        if (collisionTreeDisplayLevel < 0) { collisionTreeDisplayLevel = 0; }
        cColorf color = cColorf(1.0, 0.0, 0.0);
        object->setCollisionDetectorProperties(collisionTreeDisplayLevel, color, true);
        object->setShowCollisionDetector(true, true);
    }

    // option +:
    if (key == '+')
    {
        collisionTreeDisplayLevel++;
        cColorf color = cColorf(1.0, 0.0, 0.0);
        object->setCollisionDetectorProperties(collisionTreeDisplayLevel, color, true);
        object->setShowCollisionDetector(true, true);
    }

    // option s: save screeshot to file
    if (key == 's')
    {
        cImagePtr image = cImage::create();
        camera->copyImageBuffer(image);
        image->saveToFile("screenshot.png");
        cout << "> Saved screenshot to file.       \r";
    }

    // option t: show/hide triangles
    if (key == 't')
    {
        showTriangles = !showTriangles;
        object->setShowTriangles(showTriangles);
    }

    // option e: show/hide edges
    if (key == 'e')
    {
        showEdges = !showEdges;
        object->setShowEdges(showEdges);
    }

    // option n: show/hide normals
    if (key == 'n')
    {
        showNormals = !showNormals;
        object->setShowNormals(showNormals);
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

    // compute new camera angles
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

void updateGraphics(void)
{
    /////////////////////////////////////////////////////////////////////
    // UPDATE WIDGETS
    /////////////////////////////////////////////////////////////////////

    // update haptic rate data
    labelHapticRate->setText(cStr(frequencyCounter.getFrequency(), 0) + " Hz");

    // update position of label
    labelHapticRate->setLocalPos((int)(0.5 * (windowW - labelHapticRate->getWidth())), 15);


    /////////////////////////////////////////////////////////////////////
    // RENDER SCENE
    /////////////////////////////////////////////////////////////////////

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
    simulationRunning  = true;
    simulationFinished = false;

	cCollisionRecorder recorder;
	cCollisionSettings settings;
	
	cVector3d prev_pos = tool->getGlobalPos();
    // main haptic simulation loop
    while(simulationRunning)
    {
        /////////////////////////////////////////////////////////////////////////
        // HAPTIC RENDERING
        /////////////////////////////////////////////////////////////////////////

        // update frequency counter
        frequencyCounter.signal(1);
		tool->initialize();
        // compute global reference frames for each object
        world->computeGlobalPositions(true);

        // update position and orientation of tool
        tool->updateFromDevice();

        // compute interaction forces
        tool->computeInteractionForces();
		cVector3d position;
		hapticDevice->getPosition(position);
		
		cVector3d curr_pos = position;

		//cout << curr_pos << endl;
		//cout << prev_pos << endl;

		//bool hit = world->computeCollisionDetection(curr_pos,
		//	prev_pos,
		//	recorder,
		//	settings);
		//if (hit)
		//{

		//	cout << "collision" << endl;

		//}

		//else if (!hit)
		//{
		//	//cout << "not collsioin" << endl;
		//}

		//if (tool->m_hapticPoint->getNumCollisionEvents() > 0)
		//{

		//	cout << "collision" << endl;

		//}

		//else if (tool->m_hapticPoint->getNumCollisionEvents() == 0)
		//{
		//	cout << "not collsioin" << endl;
		//}

		prev_pos = curr_pos;

		//bool collision = CollisionCheck(object, tool);
		//if (collision)
		//{
		//	cout << "custom collision" << endl;
		//}
		//else
		//{
		//	cout << "not collision" << endl;
		//}

		
 
        /////////////////////////////////////////////////////////////////////////
        // MANIPULATION
        /////////////////////////////////////////////////////////////////////////

        // compute transformation from world to tool (haptic device)
        cTransform world_T_tool = tool->getDeviceGlobalTransform();

        // get status of user switch
        bool button = tool->getUserSwitch(0);

        //
        // STATE 1:
        // Idle mode - user presses the user switch
        //
        if ((state == IDLE) && (button == true))
        {
            // check if at least one contact has occurred
            if (tool->m_hapticPoint->getNumCollisionEvents() > 0)
            {
                // get contact event
                cCollisionEvent* collisionEvent = tool->m_hapticPoint->getCollisionEvent(0);

                // get object from contact event
                selectedObject = collisionEvent->m_object;
            }
            else
            {
                selectedObject = object;
            }

            // get transformation from object
            cTransform world_T_object = selectedObject->getGlobalTransform();

            // compute inverse transformation from contact point to object 
            cTransform tool_T_world = world_T_tool;
            tool_T_world.invert();

            // store current transformation tool
            tool_T_object = tool_T_world * world_T_object;

            // update state
            state = SELECTION;
        }


        //
        // STATE 2:
        // Selection mode - operator maintains user switch enabled and moves object
        //
        else if ((state == SELECTION) && (button == true))
        {
            // compute new transformation of object in global coordinates
            cTransform world_T_object = world_T_tool * tool_T_object;

            // compute new transformation of object in local coordinates
            cTransform parent_T_world = selectedObject->getParent()->getLocalTransform();
            parent_T_world.invert();
            cTransform parent_T_object = parent_T_world * world_T_object;

            // assign new local transformation to object
            selectedObject->setLocalTransform(parent_T_object);

            // set zero forces when manipulating objects
            tool->setDeviceGlobalForce(0.0, 0.0, 0.0);

            tool->initialize();
        }

        //
        // STATE 3:
        // Finalize Selection mode - operator releases user switch.
        //
        else
        {
            state = IDLE;
        }


        /////////////////////////////////////////////////////////////////////////
        // FINALIZE
        /////////////////////////////////////////////////////////////////////////

        // send forces to haptic device
        tool->applyToDevice();  
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
bool CollisionCheck(cGenericObject* voxObject, cToolCursor* tool_cursor)
{

	//Get center point tool
	cVector3d center = tool_cursor->getDeviceGlobalPos();


	//cout << "tool center:  " << center << endl;
	// Calculate AABB infor (center, half-extents)
	cVector3d aabb_center = voxObject->getBoundaryCenter();
	cTransform Object_transform = voxObject->getLocalTransform();
	cMatrix3d Object_rot = voxObject->getLocalRot();

	//cout << "object center: " << aabb_center << endl;
	// Get difference vector between both centers
	cVector3d difference = center - aabb_center;

	cVector3d clamped = cVec3Clamp(difference, voxObject->getBoundaryMin(), voxObject->getBoundaryMax());

	cVector3d closest = aabb_center + clamped;


	difference = closest - center;

	return  difference.length() < tool_cursor->m_hapticPoint->getRadiusContact();
}

//------------------------------------------------------------------------------
