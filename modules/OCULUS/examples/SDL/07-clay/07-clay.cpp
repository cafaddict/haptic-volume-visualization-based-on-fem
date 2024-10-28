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
    \version   3.1.1 $Rev: 1659 $
*/
//==============================================================================

//------------------------------------------------------------------------------
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//------------------------------------------------------------------------------
#include "chai3d.h"
//------------------------------------------------------------------------------
#include "SDL.h"
#include "OVRRenderContext.h"
#include "OVRDevice.h"
//------------------------------------------------------------------------------
using namespace chai3d;
using namespace std;
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// CHAI3D
//------------------------------------------------------------------------------

// a world that contains all objects of the virtual environment
cWorld* world;

// a camera to render the world in the window display
cCamera* camera;

// a light source
cSpotLight *light;

// a haptic device handler
cHapticDeviceHandler* handler;

// a pointer to the current haptic device
cGenericHapticDevicePtr hapticDevice;

// a virtual tool representing the haptic device in the scene
cToolCursor* tool;

// a virtual mesh like object
cVoxelObject* object;

// 3D texture
cTexture3dPtr texture;

// 3D image data
cMultiImagePtr image;

// Resolution of voxel model
int voxelModelResolution = 128;

// Region of voxels being updated
cCollisionAABBBox volumeUpdate;

// indicates if the haptic simulation currently running
bool simulationRunning = false;

// indicates if the haptic simulation has terminated
bool simulationFinished = false;

// frequency counter to measure the simulation haptic rate
cFrequencyCounter frequencyCounter;

// Flag that indicates that voxels have been updated
bool flagMarkVolumeForUpdate = false;

// Mutex to voxel
cMutex mutexVoxel;

// Mutex to object
cMutex mutexObject;


//------------------------------------------------------------------------------
// OCULUS RIFT
//------------------------------------------------------------------------------

// display context
cOVRRenderContext renderContext;

// oculus device
cOVRDevice oculusVR;


//------------------------------------------------------------------------------
// DECLARED MACROS
//------------------------------------------------------------------------------

// convert to resource path
#define RESOURCE_PATH(p)    (char*)((resourceRoot+string(p)).c_str())


//------------------------------------------------------------------------------
// DECLARED FUNCTIONS
//------------------------------------------------------------------------------

// function that closes the application
void close(void);

// main haptics simulation loop
void updateHaptics(void);

// process keyboard events
void processEvents();

// build voxel shape
void buildVoxelShape(double a_radiusSphere, double a_radiusCylinder);


//==============================================================================
/*
    DEMO:   07-clay.cpp

    This example illustrates the use of voxel data to simulate a clay material
*/
//==============================================================================

int main(int argc, char **argv)
{
    //--------------------------------------------------------------------------
    // INITIALIZATION
    //--------------------------------------------------------------------------

    cout << endl;
    cout << "-----------------------------------" << endl;
    cout << "CHAI3D" << endl;
    cout << "Demo: 07-clay" << endl;
    cout << "Copyright 2003-2016" << endl;
    cout << "-----------------------------------" << endl << endl << endl;
    cout << "Keyboard Options:" << endl << endl;
    cout << "[f] - Enable/Disable full screen mode" << endl;
    cout << "[q] - Exit application" << endl;
    cout << endl << endl;

    // parse first arg to try and locate resources
    string resourceRoot = string(argv[0]).substr(0, string(argv[0]).find_last_of("/\\") + 1);


    //--------------------------------------------------------------------------
    // SETUP DISPLAY CONTEXT
    //--------------------------------------------------------------------------

    // initialize everything
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS) < 0)
    {
        cout << "failed initialization" << endl;
        cSleepMs(1000);
        return 1;
    }

    if (!oculusVR.initVR())
    {
        cout << "failed to initialize Oculus" << endl;
        cSleepMs(1000);
        SDL_Quit();
        return 1;
    }

    ovrSizei hmdResolution = oculusVR.getResolution();
    ovrSizei windowSize = { hmdResolution.w / 2, hmdResolution.h / 2 };

    renderContext.init("CHAI3D", 100, 100, windowSize.w, windowSize.h);
    SDL_ShowCursor(SDL_DISABLE);

    if (glewInit() != GLEW_OK)
    {
        oculusVR.destroyVR();
        renderContext.destroy();
        SDL_Quit();
        return 1;
    }

    if (!oculusVR.initVRBuffers(windowSize.w, windowSize.h))
    {
        oculusVR.destroyVR();
        renderContext.destroy();
        SDL_Quit();
        return 1;
    }


    //--------------------------------------------------------------------------
    // WORLD - CAMERA - LIGHTING
    //--------------------------------------------------------------------------

    // create a new world.
    world = new cWorld();

    // set the background color of the environment
    // the color is defined by its (R,G,B) components.
    world->m_backgroundColor.setWhite();

    // create a camera and insert it into the virtual world
    camera = new cCamera(world);
    world->addChild(camera);

    // position and orient the camera
    camera->set(cVector3d(0.6, 0.0, 0.4),    // camera position (eye)
                cVector3d(0.0, 0.0, 0.0),    // lookat position (target)
                cVector3d(0.0, 0.0, 1.0));   // direction of the "up" vector

    // set the near and far clipping planes of the camera
    // anything in front/behind these clipping planes will not be rendered
    camera->setClippingPlanes(0.01, 10.0);

    // create a light source
    light = new cSpotLight(world);

    // add light to world
    world->addChild(light);

    // enable light source
    light->setEnabled(true);

    // position the light source
    light->setLocalPos(3.5, 2.0, 0.0);

    // define the direction of the light beam
    light->setDir(-3.5, -2.0, 0.0);

    // set light cone half angle
    light->setCutOffAngleDeg(50);


    //--------------------------------------------------------------------------
    // HAPTIC DEVICES / TOOLS
    //--------------------------------------------------------------------------

    // create a haptic device handler
    handler = new cHapticDeviceHandler();

    // get access to the first available haptic device
    handler->getDevice(hapticDevice, 0);

    // retrieve information about the current haptic device
    cHapticDeviceInfo hapticDeviceInfo = hapticDevice->getSpecifications();

    // create a 3D tool and add it to the world
    tool = new cToolCursor(world);
    world->addChild(tool);

    // connect the haptic device to the tool
    tool->setHapticDevice(hapticDevice);

    // define the radius of the tool (sphere)
    double toolRadius = 0.025;

    // define a radius for the tool
    tool->setRadius(toolRadius);

    // enable if objects in the scene are going to rotate of translate
    // or possibly collide against the tool. If the environment
    // is entirely static, you can set this parameter to "false"
    tool->enableDynamicObjects(true);

    // map the physical workspace of the haptic device to a larger virtual workspace.
    tool->setWorkspaceRadius(0.6);

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
    double maxStiffness	= hapticDeviceInfo.m_maxLinearStiffness / workspaceScaleFactor;


    //--------------------------------------------------------------------------
    // CREATE ENVIRONMENT GLOBE
    //--------------------------------------------------------------------------

    // create a virtual mesh
    cMesh* globe = new cMesh();

    // add object to world
    world->addChild(globe);

    // set the position of the object at the center of the world
    globe->setLocalPos(0.0, 0.0, 0.0);

    // create cube
    cCreateSphere(globe, 10.0, 36, 36);

    globe->setUseDisplayList(true);
    globe->deleteCollisionDetector();

    // create a texture
    cTexture2dPtr textureSpace = cTexture2d::create();

    bool fileload = textureSpace->loadFromFile(RESOURCE_PATH("../resources/images/shade.jpg"));
    if (!fileload)
    {
#if defined(_MSVC)
        fileload = textureSpace->loadFromFile("../../../bin/resources/images/shade.jpg");
#endif
    }
    if (!fileload)
    {
        cout << "Error - Texture image failed to load correctly." << endl;
        close();
        return (-1);
    }

    // apply texture to object
    globe->setTexture(textureSpace);

    // enable texture rendering 
    globe->setUseTexture(true);

    // Since we don't need to see our polygons from both sides, we enable culling.
    globe->setUseCulling(false);

    // disable material properties and lighting
    globe->setUseMaterial(false);
    

    //--------------------------------------------------------------------------
    // CREATE VOXEL OBJECT
    //--------------------------------------------------------------------------

    // create a volumetric model
    object = new cVoxelObject();

    // add object to world
    world->addChild(object);
    
    // rotate object
    object->rotateAboutGlobalAxisDeg(0, 0, 1, 40);
    
    // set the dimensions by assigning the position of the min and max corners
    object->m_minCorner.set(-0.3,-0.3,-0.3);
    object->m_maxCorner.set( 0.3, 0.3, 0.3);

    // set the texture coordinate at each corner.
    object->m_minTextureCoord.set(0.0, 0.0, 0.0);
    object->m_maxTextureCoord.set(1.0, 1.0, 1.0);

    // set material color
    object->m_material->setOrangeCoral();

    // set stiffness property
    object->setStiffness(0.2 * maxStiffness);

    // show/hide boundary box
    object->setShowBoundaryBox(false);


    //--------------------------------------------------------------------------
    // CREATE VOXEL DATA
    //--------------------------------------------------------------------------

    // create multi image data structure
    image = cMultiImage::create();

    // allocate 3D image data
    image->allocate(voxelModelResolution, voxelModelResolution, voxelModelResolution, GL_RGBA);

    // create texture
    texture = cTexture3d::create();

    // assign texture to voxel object
    object->setTexture(texture);

    // assign volumetric image to texture
    texture->setImage(image);

    // draw some 3D volumetric object
    buildVoxelShape(0.5, 0.2);

    // set default rendering mode
    object->setRenderingModeIsosurfaceMaterial();

    // set quality of graphic rendering
    object->setQuality(0.5);


    //--------------------------------------------------------------------------
    // START SIMULATION
    //--------------------------------------------------------------------------

    // create a thread which starts the main haptics rendering loop
    cThread* hapticsThread = new cThread();
    hapticsThread->start(updateHaptics, CTHREAD_PRIORITY_HAPTICS);

    // setup callback when application exits
    atexit(close);


    //--------------------------------------------------------------------------
    // MAIN GRAPHIC LOOP
    //--------------------------------------------------------------------------
    
    while (!simulationFinished)
    {
        // handle key presses
        processEvents();

        // start rendering
        oculusVR.onRenderStart();

        // render frame for each eye
        for (int eyeIndex = 0; eyeIndex < ovrEye_Count; eyeIndex++)
        {
            // retrieve projection and modelview matrix from oculus
            cTransform projectionMatrix, modelViewMatrix;
            oculusVR.onEyeRender(eyeIndex, projectionMatrix, modelViewMatrix);

            camera->m_useCustomProjectionMatrix = true;
            camera->m_projectionMatrix = projectionMatrix;

            camera->m_useCustomModelViewMatrix = true;
            camera->m_modelViewMatrix = modelViewMatrix;

            // render world
            ovrSizei size = oculusVR.getEyeTextureSize(eyeIndex);
            camera->renderView(size.w, size.h, 0, C_STEREO_LEFT_EYE, false);

            // finalize rendering  
            oculusVR.onEyeRenderFinish(eyeIndex);
        }

        // update frames
        oculusVR.submitFrame();
        oculusVR.blitMirror();
        SDL_GL_SwapWindow(renderContext.window);


        /////////////////////////////////////////////////////////////////////
        // VOLUME UPDATE
        /////////////////////////////////////////////////////////////////////

        // update region of voxels to be updated
        if (flagMarkVolumeForUpdate)
        {
            mutexVoxel.acquire();
            cVector3d min = volumeUpdate.m_min;
            cVector3d max = volumeUpdate.m_max;
            volumeUpdate.setEmpty();
            mutexVoxel.release();
            texture->markForPartialUpdate(min, max);
            flagMarkVolumeForUpdate = false;
        }
    }

    // cleanup
    oculusVR.destroyVR();
    renderContext.destroy();

    // exit SDL
    SDL_Quit();

    return (0);
}

//------------------------------------------------------------------------------

void processEvents()
{
    SDL_Event event;

    while (SDL_PollEvent(&event))
    {
        switch (event.type)
        {
        case SDL_KEYDOWN:
            // esc
            if (event.key.keysym.sym == SDLK_q || event.key.keysym.sym == SDLK_ESCAPE)
            {
                close();
                break;
            }

            // spacebar
            if (event.key.keysym.sym == SDLK_SPACE)
            {
                oculusVR.recenterPose();
                break;
            }

            break;

        case SDL_QUIT:
            close();
            break;

        default:
            break;
        }
    }
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

void updateHaptics(void)
{
    // reset clock
    cPrecisionClock clock;
    clock.reset();

    // simulation in now running
    simulationRunning  = true;
    simulationFinished = false;

    // main haptic simulation loop
    while(simulationRunning)
    {
        /////////////////////////////////////////////////////////////////////
        // SIMULATION TIME
        /////////////////////////////////////////////////////////////////////

        // stop the simulation clock
        clock.stop();

        // read the time increment in seconds
        double timeInterval = clock.getCurrentTimeSeconds();

        // restart the simulation clock
        clock.reset();
        clock.start();

        // update frequency counter
        frequencyCounter.signal(1);


        /////////////////////////////////////////////////////////////////////
        // HAPTIC FORCE COMPUTATION
        /////////////////////////////////////////////////////////////////////

        // compute global reference frames for each object
        world->computeGlobalPositions(true);

        // update position and orientation of tool
        tool->updateFromDevice();

        // read user switch
        int userSwitches = tool->getUserSwitches();

        // acquire mutex
        if (mutexObject.tryAcquire())
        {
            // compute interaction forces
            tool->computeInteractionForces();

            // check if tool is in contact with voxel object
            if (tool->isInContact(object) && (userSwitches > 0))
            {
                // retrieve contact event
                cCollisionEvent* contact = tool->m_hapticPoint->getCollisionEvent(0);

                // update voxel color
                cColorb color(0x00, 0x00, 0x00, 0x00);
                object->m_texture->m_image->setVoxelColor(contact->m_voxelIndexX, contact->m_voxelIndexY, contact->m_voxelIndexZ, color);

                // mark voxel for update
                mutexVoxel.acquire();
                volumeUpdate.enclose(cVector3d(contact->m_voxelIndexX, contact->m_voxelIndexY, contact->m_voxelIndexZ));
                mutexVoxel.release();
                flagMarkVolumeForUpdate = true;   
            }

            // release mutex
            mutexObject.release();
        }

        // send forces to haptic device
        tool->applyToDevice();
    }
    
    // exit haptics thread
    simulationFinished = true;
}

//------------------------------------------------------------------------------

void buildVoxelShape(double a_radiusSphere, double a_radiusCylinder)
{
    // setup dimension of shape in voxel resolution
    double center = (double)voxelModelResolution / 2.0;
    double radiusSphere = (double)voxelModelResolution * a_radiusSphere;
    double radiusCylinder = (double)voxelModelResolution * a_radiusCylinder;
    double k = 255.0 / (double)voxelModelResolution;
    
    // draw voxels
    for (int z=0; z<voxelModelResolution; z++)
    {
        for (int y=0; y<voxelModelResolution; y++)
        {
            for (int x=0; x<voxelModelResolution; x++)
            {
                double px = (double)x;
                double py = (double)y;
                double pz = (double)z;

                // set translucent voxel
                cColorb color;
                color.set(0x00, 0x00, 0x00, 0x00);

                double distance = sqrt(cSqr(px-center) + cSqr(py-center) + cSqr(pz-center));
                if (distance < radiusSphere)
                {
                    double distanceAxisZ = sqrt(cSqr(px-center) + cSqr(py-center));
                    double distanceAxisY = sqrt(cSqr(px-center) + cSqr(pz-center));
                    double distanceAxisX = sqrt(cSqr(py-center) + cSqr(pz-center));
                
                    if ((distanceAxisZ > radiusCylinder) && (distanceAxisY > radiusCylinder) && (distanceAxisX > radiusCylinder))
                    {
                        // assign color to voxel
                        //color.set(0xff, 0xff, 0xff, 0xff);
                        color.set((GLubyte)(k * fabs(px-center)), (GLubyte)(k * fabs(py-center)), (GLubyte)(k * fabs(pz-center)), 0xff);
                    }
                }

                image->setVoxelColor(x, y, z, color);
            }
        }
    }

    texture->markForUpdate();
}

//------------------------------------------------------------------------------