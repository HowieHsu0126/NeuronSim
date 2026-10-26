/* * Copyright (c) 2016 Regents of the University of California. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*
* 1. Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*
* 2. Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
*
* 3. The names of its contributors may not be used to endorse or promote
*    products derived from this software without specific prior written
*    permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
* A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
* LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
* NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* *********************************************************************************************** *
* CARLsim
* created by: (MDR) Micah Richert, (JN) Jayram M. Nageswaran
* maintained by:
* (MA) Mike Avery <averym@uci.edu>
* (MB) Michael Beyeler <mbeyeler@uci.edu>,
* (KDC) Kristofor Carlson <kdcarlso@uci.edu>
* (TSC) Ting-Shuo Chou <tingshuc@uci.edu>
* (HK) Hirak J Kashyap <kashyaph@uci.edu>
*
* CARLsim v1.0: JM, MDR
* CARLsim v2.0/v2.1/v2.2: JM, MDR, MA, MB, KDC
* CARLsim3: MB, KDC, TSC
* CARLsim4: TSC, HK
*
* CARLsim available from http://socsci.uci.edu/~jkrichma/CARLsim/
* Ver 12/31/2016
*/

// include CARLsim user interface
#include <carlsim.h>

// Include libraries that will allow for us to perform vector operations, and
// print their results
#include <iostream>
#include <random>
#include <iterator>
#include <fstream>
#include <algorithm>
#include <vector>
#include <boost/iterator/counting_iterator.hpp>
#include <ctime>
#include <cstdlib>

using namespace std;

// include stopwatch for timing
#include <stopwatch.h>

// Create a function that will print out all of the elements of a vector
void print(std::vector <int> const &a) {
 std::cout << "The vector elements are : ";

 for(int i=0; i < a.size(); i++)
		std::cout << a.at(i) << ' ';
}

// Create a function that will create a subset of a vector, which will can be
// used in defining Poisson rates for a fraction of neurons in a group
template<typename T>
std::vector<T> slice(std::vector<T> &v, int m, int n)
{
    std::vector<T> vec;
    std::copy(v.begin() + m, v.begin() + n + 1, std::back_inserter(vec));
    return vec;
}

// std::default_random_engine exc_generator;
// std::default_random_engine inh_generator;
std::default_random_engine pyr_generator;
std::default_random_engine pyr_generator_recall;
std::default_random_engine aac_generator;
std::default_random_engine bc_generator;
std::default_random_engine cck_generator;
std::default_random_engine bic_generator;
std::default_random_engine ivy_generator;
std::default_random_engine mfao_generator;
std::default_random_engine quadD_generator;
// std::lognormal_distribution<float> distribution(-4.0,1.0);
// std::lognormal_distribution<float> exc_distribution(4.0,1.5);
// std::lognormal_distribution<float> inh_distribution(4.0,1.25);
std::lognormal_distribution<float> pyr_distribution(4.0,1.5);
std::lognormal_distribution<float> pyr_distribution_recall(4.0,1.0);
std::lognormal_distribution<float> aac_distribution(4.0,1.25);
std::lognormal_distribution<float> bc_distribution(4.0,1.75);
std::lognormal_distribution<float> cck_distribution(4.0,1.0);
std::lognormal_distribution<float> bic_distribution(4.0,1.0);
std::lognormal_distribution<float> ivy_distribution(4.0,1.75);
std::lognormal_distribution<float> mfao_distribution(4.0,1.25);
std::lognormal_distribution<float> quadD_distribution(4.0,1.25);

// // custom ConnectionGenerator
// class MyConnection : public ConnectionGenerator {
// public:
//     MyConnection() {}
//     ~MyConnection() {}
 
//     // the pure virtual function inherited from base class
//     // note that weight, maxWt, delay, and connected are passed by reference
//     void connect(CARLsim* sim, int srcGrp, int i, int destGrp, int j, float& weight, float& maxWt,
//             float& delay, bool& connected) {
//         // connect n-th neuron in pre to the neuron in post (with 20% prob)
//         connected = i && (drand48() < 0.0250664662231983f);
//         if ((i >= 0 && i < 300) && (j >= 0 && j < 300))
//         {
//             weight = 25.0f;
//             maxWt = 25.0f;
//             delay = 1;
//         }
        
//         else if ((i >= 300 && i < 600) && (j >= 300 && j < 600))
//         {
//             weight = 25.0f;
//             maxWt = 25.0f;
//             delay = 1;
//         }

//         else if ((i >= 600 && i < 900) && (j >= 600 && j < 900))
//         {
//             weight = 25.0f;
//             maxWt = 25.0f;
//             delay = 1;
//         }
        
//         else
//         {
//             weight = 0.5f;
//             // weight = distribution(generator);
//             maxWt = 25.0f;
//             delay = 1;
//         }
//     }
// };

int main() {
	// keep track of execution time
	Stopwatch watch;

	// ---------------- CONFIG STATE -------------------

	// create a network on GPU
	int numGPUs = 0;
	int randSeed = 10;
	CARLsim sim("ca3_snn_GPU", GPU_MODE, SILENT, numGPUs, randSeed);

	// include header file that contains generation of groups and their
	// properties
	#include "../generateCONFIGStateSTP.h"

    // Read in beta peak pattern
    std::ifstream is("trainPat.csv");
    std::istream_iterator<int> start(is), end;
    std::vector<int> trainPat(start, end);
    
	// Set the time constants for the excitatory and inhibitory receptors, and
	// set the method of integration to numerically solve the systems of ODEs
	// involved in the SNN
	// sim.setConductances(true);
	sim.setIntegrationMethod(RUNGE_KUTTA4, 5);

    // before calling setupNetwork, call loadSimulation
    FILE* fId = NULL;
    // fId = fopen("networkA.dat", "rb");
    // fId = fopen("networkZ.dat", "rb");
    fId = fopen("/scratch/jkopsick/git_09_05_23/CARLsim6/.build/projects/ca3_full_net_lognormalI_09_05_23_assem_275/network65p_assem275_divnorm.dat", "rb");
    sim.loadSimulation(fId);
    
	// ---------------- SETUP STATE -------------------
	// build the network
	watch.lap("setupNetwork");
	sim.setupNetwork();

    // ... wait until after setupNetwork is called
    fclose(fId);

	// Declare variables that will store the start and end ID for the neurons
	// in the pyramidal group
	int pyr_start = sim.getGroupStartNeuronId(CA3_Pyramidal);
	int pyr_end = sim.getGroupEndNeuronId(CA3_Pyramidal);
	int pyr_range = (pyr_end - pyr_start) + 1;

	int aac_start = sim.getGroupStartNeuronId(CA3_Axo_Axonic);
	int aac_end = sim.getGroupEndNeuronId(CA3_Axo_Axonic);
	int aac_range = (aac_end - aac_start) + 1;

	int bc_start = sim.getGroupStartNeuronId(CA3_Basket);
	int bc_end = sim.getGroupEndNeuronId(CA3_Basket);
	int bc_range = (bc_end - bc_start) + 1;

	int cck_start = sim.getGroupStartNeuronId(CA3_BC_CCK);
	int cck_end = sim.getGroupEndNeuronId(CA3_BC_CCK);
	int cck_range = (cck_end - cck_start) + 1;

	int bic_start = sim.getGroupStartNeuronId(CA3_Bistratified);
	int bic_end = sim.getGroupEndNeuronId(CA3_Bistratified);
	int bic_range = (bic_end - bic_start) + 1;
    
	int ivy_start = sim.getGroupStartNeuronId(CA3_Ivy);
	int ivy_end = sim.getGroupEndNeuronId(CA3_Ivy);
	int ivy_range = (ivy_end - ivy_start) + 1;
    
	int mfao_start = sim.getGroupStartNeuronId(CA3_MFA_ORDEN);
	int mfao_end = sim.getGroupEndNeuronId(CA3_MFA_ORDEN);
	int mfao_range = (mfao_end - mfao_start) + 1;

	int quad_start = sim.getGroupStartNeuronId(CA3_QuadD_LM);
	int quad_end = sim.getGroupEndNeuronId(CA3_QuadD_LM);
	int quad_range = (quad_end - quad_start) + 1;
    
	// Create vectors that are the length of the number of neurons in the pyramidal
	// group, and another that will store the current at the position for the
  // random pyramidal cells that will be selected
    std::vector<int> pyr_vec( boost::counting_iterator<int>( 0 ),
                              boost::counting_iterator<int>( pyr_range ));
    std::vector<float> pc_background_current(pyr_range, 0.0f);
    std::vector<float> pc_current = pc_background_current;

    std::vector<int> aac_vec( boost::counting_iterator<int>( 0 ),
                              boost::counting_iterator<int>( aac_range ));
    std::vector<float> aac_background_current(aac_range, 0.0f);
    std::vector<float> aac_current = aac_background_current;
    
    std::vector<int> bc_vec( boost::counting_iterator<int>( 0 ),
                              boost::counting_iterator<int>( bc_range ));
    std::vector<float> bc_background_current(bc_range, 0.0f);
    std::vector<float> bc_current = bc_background_current;
    
    std::vector<int> cck_vec( boost::counting_iterator<int>( 0 ),
                              boost::counting_iterator<int>( cck_range ));
    std::vector<float> cck_background_current(cck_range, 0.0f);
    std::vector<float> cck_current = cck_background_current;
    
    std::vector<int> bic_vec( boost::counting_iterator<int>( 0 ),
                              boost::counting_iterator<int>( bic_range ));
    std::vector<float> bic_background_current(bic_range, 0.0f);
    std::vector<float> bic_current = bic_background_current;
    
    std::vector<int> ivy_vec( boost::counting_iterator<int>( 0 ),
                              boost::counting_iterator<int>( ivy_range ));
    std::vector<float> ivy_background_current(ivy_range, 0.0f);
    std::vector<float> ivy_current = ivy_background_current;
    
    std::vector<int> mfao_vec( boost::counting_iterator<int>( 0 ),
                              boost::counting_iterator<int>( mfao_range ));
    std::vector<float> mfao_background_current(mfao_range, 0.0f);
    std::vector<float> mfao_current = mfao_background_current;
    
    std::vector<int> quad_vec( boost::counting_iterator<int>( 0 ),
                              boost::counting_iterator<int>( quad_range ));
    std::vector<float> quad_background_current(quad_range, 0.0f);
    std::vector<float> quad_current = quad_background_current;
    
//     float pc_Iapp = 50.0f;
//     float pc_Iapp_sd = 200.0f;
    
//     for (int i = 0; i < pyr_range; i++)
//     {
//         pc_background_current.at(i) = pc_Iapp + pc_Iapp_sd * (float)drand48();
//     }   

//     float aac_Iapp = 10.0f;
//     float aac_Iapp_sd = 50.0f;
//     for (int i = 0; i < aac_range; i++)
//     {
//         aac_background_current.at(i) = aac_Iapp + aac_Iapp_sd * (float)drand48();
//     } 

//     float bc_Iapp = 100.0f;
//     float bc_Iapp_sd = 200.0f;
//     for (int i = 0; i < bc_range; i++)
//     {
//         bc_background_current.at(i) = bc_Iapp + bc_Iapp_sd * (float)drand48();
//     }

//     float cck_Iapp = 10.0f;
//     float cck_Iapp_sd = 50.0f;
//     for (int i = 0; i < cck_range; i++)
//     {
//         cck_background_current.at(i) = cck_Iapp + cck_Iapp_sd * (float)drand48();
//     }

//     float bic_Iapp = 10.0f;
//     float bic_Iapp_sd = 50.0f;
//     for (int i = 0; i < bic_range; i++)
//     {
//         bic_background_current.at(i) = bic_Iapp + bic_Iapp_sd * (float)drand48();
//     }

//     float ivy_Iapp = 100.0f;
//     float ivy_Iapp_sd = 200.0f;
//     for (int i = 0; i < ivy_range; i++)
//     {
//         ivy_background_current.at(i) = ivy_Iapp + ivy_Iapp_sd * (float)drand48();
//     }

//     float mfao_Iapp = 100.0f;
//     float mfao_Iapp_sd = 200.0f;
//     for (int i = 0; i < mfao_range; i++)
//     {
//         mfao_background_current.at(i) = mfao_Iapp + mfao_Iapp_sd * (float)drand48();
//     }

//     float quad_Iapp = 10.0f;
//     float quad_Iapp_sd = 50.0f;
//     for (int i = 0; i < quad_range; i++)
//     {
//         quad_background_current.at(i) = quad_Iapp + quad_Iapp_sd * (float)drand48();
//     }
    
    // Define the number of neurons to receive input from the external current
    int numPyramidalFire = 275;
    int k = 0;
    int m = 300;
    int n = 300;

    std::vector<int> pyr_vec_A;
    // Define the location of those granule and stellate cells so that we choose the same granule and stellate cells
    for (int i = k; i < numPyramidalFire; i++)
    {
        pyr_vec_A.push_back(1*(i+1)-1);
    } 
    
	// include header file that contains generation of groups and their
	// properties
	#include "../generateSETUPStateSTP.h"

	// Set the seed of the pseudo-random number generator based on the current system time
// 	std::srand(std::time(nullptr));
    int myrandom = 3;
    std::srand(myrandom);

    // Define the vectors to be used for storing the random cells of each neuron type chosen
//     std::random_shuffle(&pyr_vec_A[0], &pyr_vec_A[300]);
//     print(pyr_vec_A);
    // Define vectors to store the random neurons in the assembly to be activated irregularly,
    // and a counting variable
    std::vector<int> v;
    std::vector<int> w;
    int g = 0;
    int gammaCycle = 20;
    int numIrrG = trainPat.size()/gammaCycle;
    
    for (int i=0; i < gammaCycle; i++)
    {
        for (int j = g; j < g + numIrrG; j++)
        {
            v.push_back(pyr_vec_A[trainPat[j]]);
        }
        g += numIrrG;
    }
    g = 0;
    
	// Set the seed of the pseudo-random number generator based on the current system time
// 	std::srand(std::time(nullptr));

	// ---------------- RUN STATE -------------------
	watch.lap("runNetwork");

    // save initial network
    // sim.saveSimulation("networkA.dat", true); // fileName, saveSynapseInfo 
    
	// run for a total of 10 seconds
	// at the end of each runNetwork call, SpikeMonitor stats will be printed
// 	for (int i=0; i<21; i++) 
//     {
        
//         if (i >= 0 && i < 12)
//         {
//             sim.setExternalCurrent(CA3_Pyramidal, pc_background_current);
//             sim.setExternalCurrent(CA3_Basket, bc_background_current);
//             sim.runNetwork(0,500);
//         }
        
//         if (i >= 12 && i < 18)
//         {
//             if (i % 2 == 0)
//             {
                
//                 for (int i = 0; i < gammaCycle; i++)
//                 {
//                     for (int j = g; j < g + numIrrG; j++)
//                     {
//                         int randPyrCell = pyr_vec.front() + v[j];
//                         if (randPyrCell < k + 270)
//                         {
//                             std::cout << "PC chosen is: " << randPyrCell << "\n";
//                             pc_current.at(randPyrCell) = 40000.0f;
//                         }
//                     }
                    
//                     sim.setExternalCurrent(CA3_Pyramidal, pc_current);
//                     sim.runNetwork(0,1);
                    
//                     for (int j = g; j < g + numIrrG; j++)
//                     {
//                         int randPyrCell = pyr_vec.front() + v[j];
// //                         std::cout << "PC chosen is: " << randPyrCell << "\n";
//                         pc_current.at(randPyrCell) = 0.0f;
//                     }
//                     sim.setExternalCurrent(CA3_Pyramidal, pc_current);
//                     g += numIrrG;
//                 }
                
//                 g = 0;
                
//                 numPyramidalFire += 300; 
//                 k += 300;
            
//                 // std::fill(pc_current.begin(), pc_current.end(), 0);
//                 pc_current = pc_background_current;
//                 pyr_vec_A.clear();
//                 // Define the location of those granule and stellate cells so that we choose the same granule and stellate cells
//                 for (int i = k; i < numPyramidalFire; i++)
//                 {
//                     pyr_vec_A.push_back(1*(i+1)-1);
//                 }
                
//                 v.clear();
                
//                 std::srand(myrandom);
                
//                 for (int i=0; i < gammaCycle; i++)
//                 {
//                     for (int j = g; j < g + numIrrG; j++)
//                     {
//                         v.push_back(pyr_vec_A[trainPat[j]]);
//                     }
//                     g += numIrrG;
//                 }
//                 g = 0;
//             }
//             else
//             {
// //                 sim.setExternalCurrent(CA3_Pyramidal, 0.0f);
//                 sim.setExternalCurrent(CA3_Pyramidal, pc_background_current);
//                 sim.setExternalCurrent(CA3_Basket, bc_background_current);
//                 sim.runNetwork(0,40);
//             }
            
//         }
        
//             if (i >= 18 && i < 21)
//             {
//             sim.runNetwork(0,500);
//             }
                
//         }
	for (int i=0; i<526; i++) 
    {
        
        if (i >= 0 && i < 260)
        {
            float pc_Iapp = 0.0f;
            // float pc_Iapp = 50.0f;
            // float pc_Iapp_sd = 200.0f;

            for (int i = 0; i < pyr_range; i++)
            {
                // pc_background_current.at(i) = pc_Iapp + pc_Iapp_sd * (float)drand48();
                // pc_background_current.at(i) = pc_Iapp + exc_distribution(exc_generator) * (float)drand48();
                pc_background_current.at(i) = pc_Iapp + pyr_distribution(pyr_generator) * (float)drand48();
                if (pc_background_current.at(i) >= 10000)
                {
                    pc_background_current.at(i) = 10000.0f;
                }
            }   

            float aac_Iapp = 0.0f;
            // float aac_Iapp = 10.0f;
            // float aac_Iapp_sd = 50.0f;
            for (int i = 0; i < aac_range; i++)
            {
                // aac_background_current.at(i) = aac_Iapp + aac_Iapp_sd * (float)drand48();
                // aac_background_current.at(i) = aac_Iapp + inh_distribution(inh_generator) * (float)drand48();
                aac_background_current.at(i) = aac_Iapp + aac_distribution(aac_generator) * (float)drand48();
                if (aac_background_current.at(i) >= 10000)
                {
                    aac_background_current.at(i) = 10000.0f;
                }
            } 

            float bc_Iapp = 0.0f;
            // float bc_Iapp = 100.0f;
            // float bc_Iapp_sd = 200.0f;
            for (int i = 0; i < bc_range; i++)
            {
                // bc_background_current.at(i) = bc_Iapp + bc_Iapp_sd * (float)drand48();
                // bc_background_current.at(i) = bc_Iapp + inh_distribution(inh_generator) * (float)drand48();
                bc_background_current.at(i) = bc_Iapp + bc_distribution(bc_generator) * (float)drand48();
                if (bc_background_current.at(i) >= 10000)
                {
                    bc_background_current.at(i) = 10000.0f;
                }
            }

            float cck_Iapp = 0.0f;
            // float cck_Iapp = 10.0f;
            // float cck_Iapp_sd = 50.0f;
            for (int i = 0; i < cck_range; i++)
            {
                // cck_background_current.at(i) = cck_Iapp + cck_Iapp_sd * (float)drand48();
                // cck_background_current.at(i) = cck_Iapp + inh_distribution(inh_generator) * (float)drand48();
                cck_background_current.at(i) = cck_Iapp + cck_distribution(cck_generator) * (float)drand48();
                if (cck_background_current.at(i) >= 10000)
                {
                    cck_background_current.at(i) = 10000.0f;
                }
            }

            float bic_Iapp = 0.0f;
            // float bic_Iapp = 10.0f;
            // float bic_Iapp_sd = 50.0f;
            for (int i = 0; i < bic_range; i++)
            {
                // bic_background_current.at(i) = bic_Iapp + bic_Iapp_sd * (float)drand48();
                // bic_background_current.at(i) = bic_Iapp + inh_distribution(inh_generator) * (float)drand48();
                bic_background_current.at(i) = bic_Iapp + bic_distribution(bic_generator) * (float)drand48();
                if (bic_background_current.at(i) >= 10000)
                {
                    bic_background_current.at(i) = 10000.0f;
                }
            }

            float ivy_Iapp = 0.0f;
            // float ivy_Iapp = 100.0f;
            // float ivy_Iapp_sd = 200.0f;
            for (int i = 0; i < ivy_range; i++)
            {
                // ivy_background_current.at(i) = ivy_Iapp + ivy_Iapp_sd * (float)drand48();
                // ivy_background_current.at(i) = ivy_Iapp + inh_distribution(inh_generator) * (float)drand48();
                ivy_background_current.at(i) = ivy_Iapp + ivy_distribution(ivy_generator) * (float)drand48();
                if (ivy_background_current.at(i) >= 10000)
                {
                    ivy_background_current.at(i) = 10000.0f;
                }
            }

            float mfao_Iapp = 0.0f;
            // float mfao_Iapp = 100.0f;
            // float mfao_Iapp_sd = 200.0f;
            for (int i = 0; i < mfao_range; i++)
            {
                // mfao_background_current.at(i) = mfao_Iapp + mfao_Iapp_sd * (float)drand48();
                // mfao_background_current.at(i) = mfao_Iapp + inh_distribution(inh_generator) * (float)drand48();
                mfao_background_current.at(i) = mfao_Iapp + mfao_distribution(mfao_generator) * (float)drand48();
                if (mfao_background_current.at(i) >= 10000)
                {
                    mfao_background_current.at(i) = 10000.0f;
                }
            }

            float quad_Iapp = 0.0f;
            // float quad_Iapp = 10.0f;
            // float quad_Iapp_sd = 50.0f;
            for (int i = 0; i < quad_range; i++)
            {
                // quad_background_current.at(i) = quad_Iapp + quad_Iapp_sd * (float)drand48();
                // quad_background_current.at(i) = quad_Iapp + inh_distribution(inh_generator) * (float)drand48();
                quad_background_current.at(i) = quad_Iapp + quadD_distribution(quadD_generator) * (float)drand48();
                if (quad_background_current.at(i) >= 10000)
                {
                    quad_background_current.at(i) = 10000.0f;
                }
            }
            sim.setExternalCurrent(CA3_Pyramidal, pc_background_current);
            sim.setExternalCurrent(CA3_Axo_Axonic, aac_background_current);
            sim.setExternalCurrent(CA3_Basket, bc_background_current);
            sim.setExternalCurrent(CA3_BC_CCK, cck_background_current);
            sim.setExternalCurrent(CA3_Bistratified, bic_background_current);
            sim.setExternalCurrent(CA3_Ivy, ivy_background_current);
            sim.setExternalCurrent(CA3_MFA_ORDEN, mfao_background_current);
            sim.setExternalCurrent(CA3_QuadD_LM, quad_background_current);
            sim.runNetwork(0,1);
        }
        
        if (i >= 260 && i < 520)
        {
            float pc_Iapp = 0.0f;
            // float pc_Iapp = 50.0f;
            // float pc_Iapp_sd = 200.0f;

            for (int i = 0; i < pyr_range; i++)
            {
                // pc_background_current.at(i) = pc_Iapp + pc_Iapp_sd * (float)drand48();
                // pc_background_current.at(i) = pc_Iapp + exc_distribution(exc_generator) * (float)drand48();
                pc_background_current.at(i) = pc_Iapp + pyr_distribution(pyr_generator) * (float)drand48();
                if (pc_background_current.at(i) >= 10000)
                {
                    pc_background_current.at(i) = 10000.0f;
                }
            }   

            float aac_Iapp = 0.0f;
            // float aac_Iapp = 10.0f;
            // float aac_Iapp_sd = 50.0f;
            for (int i = 0; i < aac_range; i++)
            {
                // aac_background_current.at(i) = aac_Iapp + aac_Iapp_sd * (float)drand48();
                // aac_background_current.at(i) = aac_Iapp + inh_distribution(inh_generator) * (float)drand48();
                aac_background_current.at(i) = aac_Iapp + aac_distribution(aac_generator) * (float)drand48();
                if (aac_background_current.at(i) >= 10000)
                {
                    aac_background_current.at(i) = 10000.0f;
                }
            } 

            float bc_Iapp = 0.0f;
            // float bc_Iapp = 100.0f;
            // float bc_Iapp_sd = 200.0f;
            for (int i = 0; i < bc_range; i++)
            {
                // bc_background_current.at(i) = bc_Iapp + bc_Iapp_sd * (float)drand48();
                // bc_background_current.at(i) = bc_Iapp + inh_distribution(inh_generator) * (float)drand48();
                bc_background_current.at(i) = bc_Iapp + bc_distribution(bc_generator) * (float)drand48();
                if (bc_background_current.at(i) >= 10000)
                {
                    bc_background_current.at(i) = 10000.0f;
                }
            }

            float cck_Iapp = 0.0f;
            // float cck_Iapp = 10.0f;
            // float cck_Iapp_sd = 50.0f;
            for (int i = 0; i < cck_range; i++)
            {
                // cck_background_current.at(i) = cck_Iapp + cck_Iapp_sd * (float)drand48();
                // cck_background_current.at(i) = cck_Iapp + inh_distribution(inh_generator) * (float)drand48();
                cck_background_current.at(i) = cck_Iapp + cck_distribution(cck_generator) * (float)drand48();
                if (cck_background_current.at(i) >= 10000)
                {
                    cck_background_current.at(i) = 10000.0f;
                }
            }

            float bic_Iapp = 0.0f;
            // float bic_Iapp = 10.0f;
            // float bic_Iapp_sd = 50.0f;
            for (int i = 0; i < bic_range; i++)
            {
                // bic_background_current.at(i) = bic_Iapp + bic_Iapp_sd * (float)drand48();
                // bic_background_current.at(i) = bic_Iapp + inh_distribution(inh_generator) * (float)drand48();
                bic_background_current.at(i) = bic_Iapp + bic_distribution(bic_generator) * (float)drand48();
                if (bic_background_current.at(i) >= 10000)
                {
                    bic_background_current.at(i) = 10000.0f;
                }
            }

            float ivy_Iapp = 0.0f;
            // float ivy_Iapp = 100.0f;
            // float ivy_Iapp_sd = 200.0f;
            for (int i = 0; i < ivy_range; i++)
            {
                // ivy_background_current.at(i) = ivy_Iapp + ivy_Iapp_sd * (float)drand48();
                // ivy_background_current.at(i) = ivy_Iapp + inh_distribution(inh_generator) * (float)drand48();
                ivy_background_current.at(i) = ivy_Iapp + ivy_distribution(ivy_generator) * (float)drand48();
                if (ivy_background_current.at(i) >= 10000)
                {
                    ivy_background_current.at(i) = 10000.0f;
                }
            }

            float mfao_Iapp = 0.0f;
            // float mfao_Iapp = 100.0f;
            // float mfao_Iapp_sd = 200.0f;
            for (int i = 0; i < mfao_range; i++)
            {
                // mfao_background_current.at(i) = mfao_Iapp + mfao_Iapp_sd * (float)drand48();
                // mfao_background_current.at(i) = mfao_Iapp + inh_distribution(inh_generator) * (float)drand48();
                mfao_background_current.at(i) = mfao_Iapp + mfao_distribution(mfao_generator) * (float)drand48();
                if (mfao_background_current.at(i) >= 10000)
                {
                    mfao_background_current.at(i) = 10000.0f;
                }
            }

            float quad_Iapp = 0.0f;
            // float quad_Iapp = 10.0f;
            // float quad_Iapp_sd = 50.0f;
            for (int i = 0; i < quad_range; i++)
            {
                // quad_background_current.at(i) = quad_Iapp + quad_Iapp_sd * (float)drand48();
                // quad_background_current.at(i) = quad_Iapp + inh_distribution(inh_generator) * (float)drand48();
                quad_background_current.at(i) = quad_Iapp + quadD_distribution(quadD_generator) * (float)drand48();
                if (quad_background_current.at(i) >= 10000)
                {
                    quad_background_current.at(i) = 10000.0f;
                }
            }
            sim.setExternalCurrent(CA3_Pyramidal, pc_background_current);
            sim.setExternalCurrent(CA3_Axo_Axonic, aac_background_current);
            sim.setExternalCurrent(CA3_Basket, bc_background_current);
            sim.setExternalCurrent(CA3_BC_CCK, cck_background_current);
            sim.setExternalCurrent(CA3_Bistratified, bic_background_current);
            sim.setExternalCurrent(CA3_Ivy, ivy_background_current);
            sim.setExternalCurrent(CA3_MFA_ORDEN, mfao_background_current);
            sim.setExternalCurrent(CA3_QuadD_LM, quad_background_current);
            sim.runNetwork(0,1);
        }
        
        if (i >= 520 && i < 525)
        {
            if (i % 2 == 0)
            {
                
                for (int i = 0; i < gammaCycle; i++)
                {
                    float pc_Iapp = 0.0f;
                    for (int j = 0; j < pyr_range; j++)
                    {
                        // pc_background_current.at(i) = pc_Iapp + pc_Iapp_sd * (float)drand48();
                        // pc_background_current.at(i) = pc_Iapp + exc_distribution(exc_generator) * (float)drand48();
                        pc_background_current.at(j) = pc_Iapp + pyr_distribution_recall(pyr_generator_recall) * (float)drand48();
                    }  
                    pc_current = pc_background_current;
                    // std::fill(pc_current.begin(), pc_current.end(), 50);
                    for (int j = g; j < g + numIrrG; j++)
                    {
                        int randPyrCell = pyr_vec.front() + v[j];
                        if (randPyrCell < k + 138)
                        {
                            // std::cout << "PC chosen is: " << randPyrCell << "\n";
                            pc_current.at(randPyrCell) = 10000.0f;
                        }
                    }
                    
                    sim.setExternalCurrent(CA3_Pyramidal, pc_current);
                    sim.runNetwork(0,1);
                    
                    for (int j = g; j < g + numIrrG; j++)
                    {
                        int randPyrCell = pyr_vec.front() + v[j];
//                         std::cout << "PC chosen is: " << randPyrCell << "\n";
                        // pc_current.at(randPyrCell) = 0.0f;
                        pc_current.at(randPyrCell) = pc_background_current.at(randPyrCell);
                    }
                    sim.setExternalCurrent(CA3_Pyramidal, pc_current);
                    g += numIrrG;
                }
                
                g = 0;
                
                numPyramidalFire += 275; 
                k += 275;
            
                // std::fill(pc_current.begin(), pc_current.end(), 0);
                pc_current = pc_background_current;
                pyr_vec_A.clear();
                // Define the location of those granule and stellate cells so that we choose the same granule and stellate cells
                for (int i = k; i < numPyramidalFire; i++)
                {
                    pyr_vec_A.push_back(1*(i+1)-1);
                }
                
                v.clear();
                
                std::srand(myrandom);
                
                for (int i=0; i < gammaCycle; i++)
                {
                    for (int j = g; j < g + numIrrG; j++)
                    {
                        v.push_back(pyr_vec_A[trainPat[j]]);
                    }
                    g += numIrrG;
                }
                g = 0;
            }
            else
            {
                for (int j = 0; j < 210; j++)
                {
                    float pc_Iapp = 0.0f;
                    // float pc_Iapp = 50.0f;
                    // float pc_Iapp_sd = 200.0f;

                    for (int i = 0; i < pyr_range; i++)
                    {
                        // pc_background_current.at(i) = pc_Iapp + pc_Iapp_sd * (float)drand48();
                        // pc_background_current.at(i) = pc_Iapp + exc_distribution(exc_generator) * (float)drand48();
                        pc_background_current.at(i) = pc_Iapp + pyr_distribution(pyr_generator) * (float)drand48();
                        if (pc_background_current.at(i) >= 10000)
                        {
                            pc_background_current.at(i) = 10000.0f;
                        }
                    }   

                    float aac_Iapp = 0.0f;
                    // float aac_Iapp = 10.0f;
                    // float aac_Iapp_sd = 50.0f;
                    for (int i = 0; i < aac_range; i++)
                    {
                        // aac_background_current.at(i) = aac_Iapp + aac_Iapp_sd * (float)drand48();
                        // aac_background_current.at(i) = aac_Iapp + inh_distribution(inh_generator) * (float)drand48();
                        aac_background_current.at(i) = aac_Iapp + aac_distribution(aac_generator) * (float)drand48();
                        if (aac_background_current.at(i) >= 10000)
                        {
                            aac_background_current.at(i) = 10000.0f;
                        }
                    } 

                    float bc_Iapp = 0.0f;
                    // float bc_Iapp = 100.0f;
                    // float bc_Iapp_sd = 200.0f;
                    for (int i = 0; i < bc_range; i++)
                    {
                        // bc_background_current.at(i) = bc_Iapp + bc_Iapp_sd * (float)drand48();
                        // bc_background_current.at(i) = bc_Iapp + inh_distribution(inh_generator) * (float)drand48();
                        bc_background_current.at(i) = bc_Iapp + bc_distribution(bc_generator) * (float)drand48();
                        if (bc_background_current.at(i) >= 10000)
                        {
                            bc_background_current.at(i) = 10000.0f;
                        }
                    }

                    float cck_Iapp = 0.0f;
                    // float cck_Iapp = 10.0f;
                    // float cck_Iapp_sd = 50.0f;
                    for (int i = 0; i < cck_range; i++)
                    {
                        // cck_background_current.at(i) = cck_Iapp + cck_Iapp_sd * (float)drand48();
                        // cck_background_current.at(i) = cck_Iapp + inh_distribution(inh_generator) * (float)drand48();
                        cck_background_current.at(i) = cck_Iapp + cck_distribution(cck_generator) * (float)drand48();
                        if (cck_background_current.at(i) >= 10000)
                        {
                            cck_background_current.at(i) = 10000.0f;
                        }
                    }

                    float bic_Iapp = 0.0f;
                    // float bic_Iapp = 10.0f;
                    // float bic_Iapp_sd = 50.0f;
                    for (int i = 0; i < bic_range; i++)
                    {
                        // bic_background_current.at(i) = bic_Iapp + bic_Iapp_sd * (float)drand48();
                        // bic_background_current.at(i) = bic_Iapp + inh_distribution(inh_generator) * (float)drand48();
                        bic_background_current.at(i) = bic_Iapp + bic_distribution(bic_generator) * (float)drand48();
                        if (bic_background_current.at(i) >= 10000)
                        {
                            bic_background_current.at(i) = 10000.0f;
                        }
                    }

                    float ivy_Iapp = 0.0f;
                    // float ivy_Iapp = 100.0f;
                    // float ivy_Iapp_sd = 200.0f;
                    for (int i = 0; i < ivy_range; i++)
                    {
                        // ivy_background_current.at(i) = ivy_Iapp + ivy_Iapp_sd * (float)drand48();
                        // ivy_background_current.at(i) = ivy_Iapp + inh_distribution(inh_generator) * (float)drand48();
                        ivy_background_current.at(i) = ivy_Iapp + ivy_distribution(ivy_generator) * (float)drand48();
                        if (ivy_background_current.at(i) >= 10000)
                        {
                            ivy_background_current.at(i) = 10000.0f;
                        }
                    }

                    float mfao_Iapp = 0.0f;
                    // float mfao_Iapp = 100.0f;
                    // float mfao_Iapp_sd = 200.0f;
                    for (int i = 0; i < mfao_range; i++)
                    {
                        // mfao_background_current.at(i) = mfao_Iapp + mfao_Iapp_sd * (float)drand48();
                        // mfao_background_current.at(i) = mfao_Iapp + inh_distribution(inh_generator) * (float)drand48();
                        mfao_background_current.at(i) = mfao_Iapp + mfao_distribution(mfao_generator) * (float)drand48();
                        if (mfao_background_current.at(i) >= 10000)
                        {
                            mfao_background_current.at(i) = 10000.0f;
                        }
                    }

                    float quad_Iapp = 0.0f;
                    // float quad_Iapp = 10.0f;
                    // float quad_Iapp_sd = 50.0f;
                    for (int i = 0; i < quad_range; i++)
                    {
                        // quad_background_current.at(i) = quad_Iapp + quad_Iapp_sd * (float)drand48();
                        // quad_background_current.at(i) = quad_Iapp + inh_distribution(inh_generator) * (float)drand48();
                        quad_background_current.at(i) = quad_Iapp + quadD_distribution(quadD_generator) * (float)drand48();
                        if (quad_background_current.at(i) >= 10000)
                        {
                            quad_background_current.at(i) = 10000.0f;
                        }
                    }
                    sim.setExternalCurrent(CA3_Pyramidal, pc_background_current);
                    sim.setExternalCurrent(CA3_Axo_Axonic, aac_background_current);
                    sim.setExternalCurrent(CA3_Basket, bc_background_current);
                    sim.setExternalCurrent(CA3_BC_CCK, cck_background_current);
                    sim.setExternalCurrent(CA3_Bistratified, bic_background_current);
                    sim.setExternalCurrent(CA3_Ivy, ivy_background_current);
                    sim.setExternalCurrent(CA3_MFA_ORDEN, mfao_background_current);
                    sim.setExternalCurrent(CA3_QuadD_LM, quad_background_current);
                    sim.runNetwork(0,1);
                }
            }
            
        }
        
        if (i >= 525 && i < 526)
        {
            float pc_Iapp = 0.0f;
            // float pc_Iapp = 50.0f;
            // float pc_Iapp_sd = 200.0f;

            for (int i = 0; i < pyr_range; i++)
            {
                // pc_background_current.at(i) = pc_Iapp + pc_Iapp_sd * (float)drand48();
                // pc_background_current.at(i) = pc_Iapp + exc_distribution(exc_generator) * (float)drand48();
                pc_background_current.at(i) = pc_Iapp + pyr_distribution(pyr_generator) * (float)drand48();
                if (pc_background_current.at(i) >= 10000)
                {
                    pc_background_current.at(i) = 10000.0f;
                }
            }   

            float aac_Iapp = 0.0f;
            // float aac_Iapp = 10.0f;
            // float aac_Iapp_sd = 50.0f;
            for (int i = 0; i < aac_range; i++)
            {
                // aac_background_current.at(i) = aac_Iapp + aac_Iapp_sd * (float)drand48();
                // aac_background_current.at(i) = aac_Iapp + inh_distribution(inh_generator) * (float)drand48();
                aac_background_current.at(i) = aac_Iapp + aac_distribution(aac_generator) * (float)drand48();
                if (aac_background_current.at(i) >= 10000)
                {
                    aac_background_current.at(i) = 10000.0f;
                }
            } 

            float bc_Iapp = 0.0f;
            // float bc_Iapp = 100.0f;
            // float bc_Iapp_sd = 200.0f;
            for (int i = 0; i < bc_range; i++)
            {
                // bc_background_current.at(i) = bc_Iapp + bc_Iapp_sd * (float)drand48();
                // bc_background_current.at(i) = bc_Iapp + inh_distribution(inh_generator) * (float)drand48();
                bc_background_current.at(i) = bc_Iapp + bc_distribution(bc_generator) * (float)drand48();
                if (bc_background_current.at(i) >= 10000)
                {
                    bc_background_current.at(i) = 10000.0f;
                }
            }

            float cck_Iapp = 0.0f;
            // float cck_Iapp = 10.0f;
            // float cck_Iapp_sd = 50.0f;
            for (int i = 0; i < cck_range; i++)
            {
                // cck_background_current.at(i) = cck_Iapp + cck_Iapp_sd * (float)drand48();
                // cck_background_current.at(i) = cck_Iapp + inh_distribution(inh_generator) * (float)drand48();
                cck_background_current.at(i) = cck_Iapp + cck_distribution(cck_generator) * (float)drand48();
                if (cck_background_current.at(i) >= 10000)
                {
                    cck_background_current.at(i) = 10000.0f;
                }
            }

            float bic_Iapp = 0.0f;
            // float bic_Iapp = 10.0f;
            // float bic_Iapp_sd = 50.0f;
            for (int i = 0; i < bic_range; i++)
            {
                // bic_background_current.at(i) = bic_Iapp + bic_Iapp_sd * (float)drand48();
                // bic_background_current.at(i) = bic_Iapp + inh_distribution(inh_generator) * (float)drand48();
                bic_background_current.at(i) = bic_Iapp + bic_distribution(bic_generator) * (float)drand48();
                if (bic_background_current.at(i) >= 10000)
                {
                    bic_background_current.at(i) = 10000.0f;
                }
            }

            float ivy_Iapp = 0.0f;
            // float ivy_Iapp = 100.0f;
            // float ivy_Iapp_sd = 200.0f;
            for (int i = 0; i < ivy_range; i++)
            {
                // ivy_background_current.at(i) = ivy_Iapp + ivy_Iapp_sd * (float)drand48();
                // ivy_background_current.at(i) = ivy_Iapp + inh_distribution(inh_generator) * (float)drand48();
                ivy_background_current.at(i) = ivy_Iapp + ivy_distribution(ivy_generator) * (float)drand48();
                if (ivy_background_current.at(i) >= 10000)
                {
                    ivy_background_current.at(i) = 10000.0f;
                }
            }

            float mfao_Iapp = 0.0f;
            // float mfao_Iapp = 100.0f;
            // float mfao_Iapp_sd = 200.0f;
            for (int i = 0; i < mfao_range; i++)
            {
                // mfao_background_current.at(i) = mfao_Iapp + mfao_Iapp_sd * (float)drand48();
                // mfao_background_current.at(i) = mfao_Iapp + inh_distribution(inh_generator) * (float)drand48();
                mfao_background_current.at(i) = mfao_Iapp + mfao_distribution(mfao_generator) * (float)drand48();
                if (mfao_background_current.at(i) >= 10000)
                {
                    mfao_background_current.at(i) = 10000.0f;
                }
            }

            float quad_Iapp = 0.0f;
            // float quad_Iapp = 10.0f;
            // float quad_Iapp_sd = 50.0f;
            for (int i = 0; i < quad_range; i++)
            {
                // quad_background_current.at(i) = quad_Iapp + quad_Iapp_sd * (float)drand48();
                // quad_background_current.at(i) = quad_Iapp + inh_distribution(inh_generator) * (float)drand48();
                quad_background_current.at(i) = quad_Iapp + quadD_distribution(quadD_generator) * (float)drand48();
                if (quad_background_current.at(i) >= 10000)
                {
                    quad_background_current.at(i) = 10000.0f;
                }
            }
            sim.setExternalCurrent(CA3_Pyramidal, pc_background_current);
            sim.setExternalCurrent(CA3_Axo_Axonic, aac_background_current);
            sim.setExternalCurrent(CA3_Basket, bc_background_current);
            sim.setExternalCurrent(CA3_BC_CCK, cck_background_current);
            sim.setExternalCurrent(CA3_Bistratified, bic_background_current);
            sim.setExternalCurrent(CA3_Ivy, ivy_background_current);
            sim.setExternalCurrent(CA3_MFA_ORDEN, mfao_background_current);
            sim.setExternalCurrent(CA3_QuadD_LM, quad_background_current);
            sim.runNetwork(0,1);
        }
                
        }        
        

    // save modified network
//     sim.saveSimulation("networkZ.dat", true); // fileName, saveSynapseInfo 
    
	// print stopwatch summary
	watch.stop();

	return 0;
}