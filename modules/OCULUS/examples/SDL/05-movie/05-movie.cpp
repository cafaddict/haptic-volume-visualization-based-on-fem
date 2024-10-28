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
cDirectionalLight *light;

// a haptic device handler
cHapticDeviceHandler* handler;

// a pointer to the current haptic device
cGenericHapticDevicePtr hapticDevice;

// a virtual tool representing the haptic device in the scene
cToolCursor* tool;

// video texture
cTextureVideoPtr videoTexture;

// video player
cVideoPtr videoPlayer;

// indicates if the haptic simulation currently running
bool simulationRunning = false;

// indicates if the haptic simulation has terminated
bool simulationFinished = false;

// frequency counter to measure the simulation haptic rate
cFrequencyCounter frequencyCounter;


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


//==============================================================================
/*
    DEMO:   05-movie.cpp

    This example illustrates how to crate a 360 degrees movie using the oculus
    rift. The movie is composed of a large texture map which is projected onto
    a globe.
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
    cout << "Demo: 05-movie" << endl;
    cout << "Copyright 2003-2016" << endl;
    cout << "-----------------------------------" << endl << endl << endl;
    cout << "Keyboard Options:" << endl << endl;
    cout << "[ ] - Recenter view point" << endl;
    cout << "[q] - Exit application" << endl;
    cout << endl << endl;

    // parse first arg to try and locate resources
    string resourceRoot = string(argv[0]).substr(0, string(argv[0]).find_last_of("/\\") + 1);


    //--------------------------------------------------------------------------
    // SETUP DISPLAY CONTEXT
    //--------------------------------------------------------------------------

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
    camera->set(cVector3d(0.1, 0.0, 0.0),    // camera position (eye)
                cVector3d(0.0, 0.0, 0.0),    // lookat position (target)
                cVector3d(0.0, 0.0, 1.0));   // direction of the "up" vector

    // set the near and far clipping planes of the camera
    // anything in front/behind these clipping planes will not be rendered
    camera->setClippingPlanes(0.01, 20.0);

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

    // get access to the first available haptic device
    handler->getDevice(hapticDevice, 0);

    // retrieve information about the current haptic device
    cHapticDeviceInfo hapticDeviceInfo = hapticDevice->getSpecifications();

    // create a 3D tool and add it to the world
    tool = new cToolCursor(world);
    camera->addChild(tool);

    // set position
    tool->setLocalPos(-1.0, 0.0, 0.0);

    // connect the haptic device to the tool
    tool->setHapticDevice(hapticDevice);

    // define the radius of the tool (sphere)
    double toolRadius = 0.02;

    // define a radius for the tool
    tool->setRadius(toolRadius);

    // assign color of tool
    tool->m_hapticPoint->m_sphereProxy->m_material->setBlueCornflower();

    // enable if objects in the scene are going to rotate of translate
    // or possibly collide against the tool. If the environment
    // is entirely static, you can set this parameter to "false"
    tool->enableDynamicObjects(true);

    // map the physical workspace of the haptic device to a larger virtual workspace.
    tool->setWorkspaceRadius(1.0);

    // haptic forces are enabled only if small forces are first sent to the device;
    // this mode avoids the force spike that occurs when the application starts when 
    // the tool is located inside an object for instance. 
    tool->setWaitForSmallForce(true);

    // start the haptic tool
    tool->start();


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
    cCreateSphere(globe, 5.0, 36, 36);
    globe->setUseDisplayList(true);
    globe->deleteCollisionDetector();

    // create a texture
    videoTexture  = cTextureVideo::create();

    // load movie files
    bool fileload = videoTexture->loadFromFile(RESOURCE_PATH("../resources/videos/panorama.ogv"));
    if (!fileload)
    {
        #if defined(_MSVC)
        fileload = videoTexture->loadFromFile("../../../bin/resources/videos/panorama.ogv");
        #endif
    }
    if (!fileload)
    {
        cout << "Error - movie file failed to load correctly." << endl;
        close();
        return (-1);
    }

    // apply texture to object
    globe->setTexture(videoTexture);

    // enable texture rendering 
    globe->setUseTexture(true);

    // Since we don't need to see our polygons from both sides, we enable culling.
    globe->setUseCulling(false);

    // disable material properties and lighting
    globe->setUseMaterial(false);

    // retrieve pointer to video player
    videoPlayer = videoTexture->m_video;

    // enable auto replay
    videoPlayer->setAutoReplay(true);

    // play video
    videoPlayer->play();


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

    // stop video
    videoPlayer->stop();

    // close haptic device
    tool->stop();
}

//------------------------------------------------------------------------------

void updateHaptics(void)
{
    // simulation in now running
    simulationRunning  = true;
    simulationFinished = false;

    // reset tool
    tool->initialize();

    // main haptic simulation loop
    while(simulationRunning)
    {
        // compute global reference frames for each object
        world->computeGlobalPositions(true);

        // update position and orientation of tool
        tool->updateFromDevice();

        // compute interaction forces
        tool->computeInteractionForces();

        // send forces to haptic device
        tool->applyToDevice();
    }

    // exit haptics thread
    simulationFinished = true;
}

//------------------------------------------------------------------------------
