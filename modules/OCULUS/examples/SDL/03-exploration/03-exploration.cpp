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
#include "CODE.h"
#include "CGenericDemo.h"
#include "CDemo1.h"
#include "CDemo2.h"
#include "CDemo3.h"
#include "CDemo4.h"
//---------------------------------------------------------------------------
using namespace chai3d;
using namespace std;
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// CHAI3D
//------------------------------------------------------------------------------

// a haptic device handler
cHapticDeviceHandler* handler;

// a pointer to the current haptic device
shared_ptr<cGenericHapticDevice> m_hapticDevice0;
shared_ptr<cGenericHapticDevice> m_hapticDevice1;

// flag to indicate if the haptic simulation currently running
bool simulationRunning = false;

// flag to indicate if the haptic simulation has terminated
bool simulationFinished = false;


//---------------------------------------------------------------------------
// DEMOS
//---------------------------------------------------------------------------

//! currently active camera
cGenericDemo* m_demo;

//! Demos
cDemo1* m_demo1;
cDemo2* m_demo2;
cDemo3* m_demo3;
cDemo4* m_demo4;


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

// initialize demos
void initDemo1();
void initDemo2();
void initDemo3();
void initDemo4();


//==============================================================================
/*
    DEMO:   03-exploration.cpp

    This example illustrates how to use ODE to create a environment which
    contains dynamic objects.
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
    cout << "Demo: 03-exploration" << endl;
    cout << "Copyright 2003-2016" << endl;
    cout << "-----------------------------------" << endl << endl << endl;
    cout << "Keyboard Options:" << endl << endl;
    cout << "[ ] - Recenter view point" << endl;
    cout << "[1] - Select Demo 1" << endl;
    cout << "[2] - Select Demo 2" << endl;
    cout << "[3] - Select Demo 3" << endl;
    cout << "[4] - Select Demo 4" << endl;
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


    //-----------------------------------------------------------------------
    // HAPTIC DEVICES 
    //-----------------------------------------------------------------------

    // create a haptic device handler
    handler = new cHapticDeviceHandler();

    // get number of haptic devices
    int numDevices = handler->getNumDevices();

    // default stiffness of scene objects
    double maxStiffness = 1000.0;

    // get access to the haptic devices found
    if (numDevices > 0)
    {
        handler->getDevice(m_hapticDevice0, 0);
        maxStiffness = cMin(maxStiffness, 0.5 * m_hapticDevice0->getSpecifications().m_maxLinearStiffness);
    }

    if (numDevices > 1)
    {
        handler->getDevice(m_hapticDevice1, 1);
        maxStiffness = cMin(maxStiffness, 0.5 * m_hapticDevice1->getSpecifications().m_maxLinearStiffness);
    }


    //-----------------------------------------------------------------------
    // DEMOS
    //-----------------------------------------------------------------------

    // setup demos
    m_demo1 = new cDemo1(resourceRoot, numDevices, m_hapticDevice0, m_hapticDevice1);
    m_demo2 = new cDemo2(resourceRoot, numDevices, m_hapticDevice0, m_hapticDevice1);
    m_demo3 = new cDemo3(resourceRoot, numDevices, m_hapticDevice0, m_hapticDevice1);
    m_demo4 = new cDemo4(resourceRoot, numDevices, m_hapticDevice0, m_hapticDevice1);

    // set object stiffness in demos
    m_demo1->setStiffness(maxStiffness);
    m_demo2->setStiffness(maxStiffness);
    m_demo3->setStiffness(maxStiffness);
    m_demo4->setStiffness(maxStiffness);

    // initialize demo 1
    initDemo1();


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

        // update shadow maps (if any)
        m_demo->m_world->updateShadowMaps(false, false);

        // start rendering
        oculusVR.onRenderStart();

        // render frame for each eye
        for (int eyeIndex = 0; eyeIndex < ovrEye_Count; eyeIndex++)
        {
            // retrieve projection and modelview matrix from oculus
            cTransform projectionMatrix, modelViewMatrix;
            oculusVR.onEyeRender(eyeIndex, projectionMatrix, modelViewMatrix);

            m_demo->m_camera->m_useCustomProjectionMatrix = true;
            m_demo->m_camera->m_projectionMatrix = projectionMatrix;

            // scale head motion
            modelViewMatrix(0,3) =  0.4 * modelViewMatrix(0,3);
            modelViewMatrix(1,3) =  0.4 * modelViewMatrix(1,3);
            modelViewMatrix(2,3) =  0.4 * modelViewMatrix(2,3);

            m_demo->m_camera->m_useCustomModelViewMatrix = true;
            m_demo->m_camera->m_modelViewMatrix = modelViewMatrix;

            // render world
            ovrSizei size = oculusVR.getEyeTextureSize(eyeIndex);

            // render view
            m_demo->m_camera->renderView(size.w, size.h, 0, C_STEREO_LEFT_EYE, false); 

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

            // demo 1
            if (event.key.keysym.sym == SDLK_1)
            {
                initDemo1();
                break;
            }

            // demo 2
            if (event.key.keysym.sym == SDLK_2)
            {
                initDemo2();
                break;
            }

            // demo 3
            if (event.key.keysym.sym == SDLK_3)
            {
                initDemo3();
                break;
            }

            // demo 4
            if (event.key.keysym.sym == SDLK_4)
            {
                initDemo4();
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

//---------------------------------------------------------------------------

void initDemo1()
{
    m_demo = m_demo1;
    m_demo->init();
}

//---------------------------------------------------------------------------

void initDemo2()
{
    m_demo = m_demo2;
    m_demo->init();
}

//---------------------------------------------------------------------------

void initDemo3()
{
    m_demo = m_demo3;
    m_demo->init();
}

//---------------------------------------------------------------------------

void initDemo4()
{
    m_demo = m_demo4;
    m_demo->init();
}

//---------------------------------------------------------------------------

void close(void)
{
    // stop the simulation
    simulationRunning = false;

    // wait for graphics and haptics loops to terminate
    while (!simulationFinished) { cSleepMs(100); }
}

//------------------------------------------------------------------------------

void updateHaptics(void)
{
    // simulation in now running
    simulationRunning = true;

    // main haptic simulation loop
    while(simulationRunning)
    {
        m_demo->updateHaptics();
    }

    // shutdown haptic devices
    if (m_demo->m_tool0 != NULL)
    {
        m_demo->m_tool0->stop();
    }
    if (m_demo->m_tool1 != NULL)
    {
        m_demo->m_tool1->stop();
    }

    // exit haptics thread
    simulationFinished = true;
}

//------------------------------------------------------------------------------
