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
#include <string>
#include <algorithm>
#include <vector>
#include <boost/iterator/counting_iterator.hpp>
#include <ctime>
#include <cstdlib>
#include <unordered_map>

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

// std::default_random_engine pyr_generator;
// std::default_random_engine pyr_generator_recall;
// std::default_random_engine aac_generator;
// std::default_random_engine bc_generator;
// std::default_random_engine cck_generator;
// std::default_random_engine bic_generator;
// std::default_random_engine ivy_generator;
// std::default_random_engine mfao_generator;
// std::default_random_engine quadD_generator;
// std::lognormal_distribution<float> pyr_distribution(4.0,1.5);
// std::lognormal_distribution<float> pyr_distribution_recall(4.0,1.0);
// std::lognormal_distribution<float> aac_distribution(4.0,1.25);
// std::lognormal_distribution<float> bc_distribution(4.0,1.75);
// std::lognormal_distribution<float> cck_distribution(4.0,1.0);
// std::lognormal_distribution<float> bic_distribution(4.0,1.0);
// std::lognormal_distribution<float> ivy_distribution(4.0,1.75);
// std::lognormal_distribution<float> mfao_distribution(4.0,1.25);
// std::lognormal_distribution<float> quadD_distribution(4.0,1.25);

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
std::lognormal_distribution<float> bc_distribution(5.5,1.0);
std::lognormal_distribution<float> cck_distribution(4.0,1.0);
std::lognormal_distribution<float> bic_distribution(4.0,1.0);
std::lognormal_distribution<float> ivy_distribution(5.5,1.0);
std::lognormal_distribution<float> mfao_distribution(4.0,1.25);
std::lognormal_distribution<float> quadD_distribution(4.0,1.25);

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

	// Set the time constants for the excitatory and inhibitory receptors, and
	// set the method of integration to numerically solve the systems of ODEs
	// involved in the SNN
	// sim.setConductances	(true, 5, 150, 6, 150);
	sim.setIntegrationMethod(RUNGE_KUTTA4, 5);

    // before calling setupNetwork, call loadSimulation
//     FILE* fId = NULL;
//     fId = fopen("networkZ.dat", "rb");
//     sim.loadSimulation(fId);
    
	// ---------------- SETUP STATE -------------------
	// build the network
	watch.lap("setupNetwork");
	sim.setupNetwork();

    // ... wait until after setupNetwork is called
//     fclose(fId);

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

    // Read in beta peak pattern
    std::ifstream is("trainPat.csv");
    std::istream_iterator<int> start(is), end;
    std::vector<int> trainPat(start, end);

	// Set the seed of the pseudo-random number generator based on the current system time
// 	std::srand(std::time(nullptr));
    int myrandom = 3;
    std::srand(myrandom);

    // Define vectors to store the random neurons in the assembly to be activated irregularly,
    // and a counting variable
    std::vector<int> v;
    std::vector<int> w;
    int g = 0;
//     int numSpikes = 2800;
    int gammaCycle = 20;
    int numIrrG = trainPat.size()/gammaCycle;
//     int numIrrG = numSpikes/gammaCycle;
    int numPatterns = 3;
    
    for (int a = 0; a < numPatterns; a++)
    {    
        for (int i=0; i < gammaCycle; i++)
        {
            
//             std::random_shuffle(&pyr_vec_A[0], &pyr_vec_A[300]);
            if (i >= gammaCycle/2)
            {            
                for (int j = g; j < g + numIrrG; j++)
                {
                    v.push_back(pyr_vec_A[trainPat[j]]);
                    w.push_back(pyr_vec_A[trainPat[j]]);
                }
            }
            else
            {
                for (int j = g; j < g + numIrrG; j++)
                {
//                     std::cout << "PC chosen is: " << pyr_vec_A[j] << "\n"; 
                    v.push_back(pyr_vec_A[trainPat[j]]);
                }
            }
            g += numIrrG;
        }
        
        
        // create an ofstream for the file output (see the link on streams for
        // more info)
        std::ofstream fs;        

        // create a name for the file output
        std::string filename = "pc_input_pat_";
        filename.append(std::to_string(a+1));
        filename.append(".csv");

        // create and open the .csv file
        fs.open(filename);
        // write the file headers
        fs << "pc" << "," << "count" << std::endl;

        const size_t v_size = v.size();
        std::unordered_map<unsigned, size_t> counts_v;
        for (auto a : v)
        {
           counts_v[a]++; 
        }

        for (auto const &p : counts_v)
        {
           // std::cout << "The value: " << p.first << " occurred " << p.second << "times\n";
           fs << p.first << "," << p.second << std::endl;
        }

        // close the output file
        fs.close();

        // create an ofstream for the file output (see the link on streams for
        // more info)
        std::ofstream fs2;        

        // create a name for the file output
        std::string filename2 = "pc_input_pat_";
        filename2.append(std::to_string(a+1));
        filename2.append("_half.csv");
        
        // create and open the .csv file
        fs2.open(filename2);
        // write the file headers
        fs2 << "pc" << "," << "count" << std::endl;

        const size_t w_size = w.size();
        std::unordered_map<unsigned, size_t> counts_w;
        for (auto a : w)
        {
           counts_w[a]++; 
        }

        for (auto const &p : counts_w)
        {
           // std::cout << "The value: " << p.first << " occurred " << p.second << "times\n";
           fs2 << p.first << "," << p.second << std::endl;
        }

        // close the output file
        fs2.close();

        // create an ofstream for the file output (see the link on streams for
        // more info)
        std::ofstream fs3;        

        // create a name for the file output
        std::string filename3 = "pc_input_pat_temp";
        filename3.append(std::to_string(a+1));
        filename3.append("_half.csv");
        
        // create and open the .csv file
        fs3.open(filename3);
        // write the file headers
        fs3 << "pc" << std::endl;

        for (int i = 0; i < w_size; i++)
        {
           fs3 << w[i] << std::endl;
        }

        // close the output file
        fs3.close();
        
        g = 0;
        numPyramidalFire += 275; 
        k += 275;
        
        pyr_vec_A.clear();

        // Define the location of those granule and stellate cells so that we choose the same granule and stellate cells
        for (int i = k; i < numPyramidalFire; i++)
        {
            pyr_vec_A.push_back(1*(i+1)-1);
        }
        
        v.clear();
        
        w.clear();
        
        std::srand(myrandom);

//         for (int i=0; i < gammaCycle; i++)
//         {
//             std::random_shuffle(&pyr_vec_A[0], &pyr_vec_A[300]);
//             for (int j = 0; j < numIrrG; j++)
//             {
// //                             std::cout << "PC chosen is: " << pyr_vec_A[j] << "\n"; 
//                 v.push_back(pyr_vec_A[j]);
//             }
//         }
    }

    
    pyr_vec_A.clear();
    g = 0;
    numPyramidalFire = 275; 
    k = 0;
    std::srand(myrandom);
    // Define the location of those granule and stellate cells so that we choose the same granule and stellate cells
    for (int i = k; i < numPyramidalFire; i++)
    {
        pyr_vec_A.push_back(1*(i+1)-1);
    }

//     std::srand(myrandom);

    for (int i=0; i < gammaCycle; i++)
    {
//         std::random_shuffle(&pyr_vec_A[0], &pyr_vec_A[300]);
        for (int j = g; j < g + numIrrG; j++)
        {
//             std::cout << "PC chosen is: " << pyr_vec_A[j] << "\n"; 
            v.push_back(pyr_vec_A[trainPat[j]]);
        }
        g += numIrrG;
    }
    
    g = 0;

	// ---------------- RUN STATE -------------------
	watch.lap("runNetwork");

    // save initial network
    // sim.saveSimulation("networkA.dat", true); // fileName, saveSynapseInfo 
    
	for (int i=0; i<1200; i++) 
    {
        
        if (i >= 0 && i < 500)
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
        
        if (i >= 500 && i < 1000)
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
        
        if (i >= 1000 && i < 1200)
        {
            if (i % 2 == 0)
            {
                for (int a = 0; a < numPatterns; a++)
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
                        for (int j = g; j < g + numIrrG; j++)
                        {
                            int randPyrCell = pyr_vec.front() + v[j];
                            // std::cout << "PC chosen is: " << randPyrCell << "\n";
                            // if (randPyrCell < k + 100)
                            // {
                                pc_current.at(randPyrCell) = 10000.0f;
                            // }
                        }

                        sim.setExternalCurrent(CA3_Pyramidal, pc_current);
                        sim.runNetwork(0,1);

                        for (int j = g; j < g + numIrrG; j++)
                        {
                            int randPyrCell = pyr_vec.front() + v[j];
//                             std::cout << "PC chosen is: " << randPyrCell << "\n";
                            pc_current.at(randPyrCell) = pc_background_current.at(randPyrCell);
                        }
                        sim.setExternalCurrent(CA3_Pyramidal, pc_current);
                        g += numIrrG;
                    }             
                    
                    g = 0;
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

//                     std::srand(myrandom);

                    for (int i=0; i < gammaCycle; i++)
                    {
//                         std::random_shuffle(&pyr_vec_A[0], &pyr_vec_A[300]);
                        for (int j = g; j < g + numIrrG; j++)
                        {
//                             std::cout << "PC chosen is: " << pyr_vec_A[j] << "\n"; 
                            v.push_back(pyr_vec_A[trainPat[j]]);
                        }
                        g += numIrrG;
                    }
                    g = 0;
                }
                g = 0;
                numPyramidalFire = 275; 
                k = 0;
                // std::fill(pc_current.begin(), pc_current.end(), 0);
                pc_current = pc_background_current;
                pyr_vec_A.clear();
                // Define the location of those granule and stellate cells so that we choose the same granule and stellate cells
                for (int i = k; i < numPyramidalFire; i++)
                {
                    pyr_vec_A.push_back(1*(i+1)-1);
                }

                v.clear();

//                 std::srand(myrandom);

                for (int i=0; i < gammaCycle; i++)
                {
//                     std::random_shuffle(&pyr_vec_A[0], &pyr_vec_A[300]);
                    for (int j = g; j < g + numIrrG; j++)
                    {
//                             std::cout << "PC chosen is: " << pyr_vec_A[j] << "\n"; 
                        v.push_back(pyr_vec_A[trainPat[j]]);
                    }
                    g += numIrrG;
                }
                g = 0;                
                
            }
            else
            {
//                 sim.setExternalCurrent(CA3_Pyramidal, 0.0f);
                sim.setExternalCurrent(CA3_Pyramidal, pc_background_current);
                sim.setExternalCurrent(CA3_Axo_Axonic, aac_background_current);
                sim.setExternalCurrent(CA3_Basket, bc_background_current);
                sim.setExternalCurrent(CA3_BC_CCK, cck_background_current);
                sim.setExternalCurrent(CA3_Bistratified, bic_background_current);
                sim.setExternalCurrent(CA3_Ivy, ivy_background_current);
                sim.setExternalCurrent(CA3_MFA_ORDEN, mfao_background_current);
                sim.setExternalCurrent(CA3_QuadD_LM, quad_background_current);
//                 sim.runNetwork(0,10);
            }

//             if (i == 4018)
//             {
//                 sim.saveSimulation("networkB.dat", true); // fileName, saveSynapseInfo
//             }
            
//             if (i == 4038)
//             {
//                 sim.saveSimulation("networkC.dat", true); // fileName, saveSynapseInfo
//             }
            
//             if (i == 4058)
//             {
//                 sim.saveSimulation("networkD.dat", true); // fileName, saveSynapseInfo
//             }
            
//             if (i == 4078)
//             {
//                 sim.saveSimulation("networkE.dat", true); // fileName, saveSynapseInfo
//             }
            
//             if (i == 4118)
//             {
//                 sim.saveSimulation("networkG.dat", true); // fileName, saveSynapseInfo
//             }
            
//             if (i == 4138)
//             {
//                 sim.saveSimulation("networkH.dat", true); // fileName, saveSynapseInfo
//             }
            
//             if (i == 4158)
//             {
//                 sim.saveSimulation("networkI.dat", true); // fileName, saveSynapseInfo
//             }
            
//             if (i == 4178)
//             {
//                 sim.saveSimulation("networkJ.dat", true); // fileName, saveSynapseInfo
//             }
            
//             if (i == 1048)
//             {
//                 sim.saveSimulation("networkB.dat", true); // fileName, saveSynapseInfo
//             }
            
//             if (i == 1098)
//             {
//                 sim.saveSimulation("networkF.dat", true); // fileName, saveSynapseInfo
//             }
            
//             if (i == 1198)
//             {
//                 sim.saveSimulation("networkZ.dat", true); // fileName, saveSynapseInfo 
//             }
            
//             if (i == 1058)
//             {
//                 sim.saveSimulation("networkC.dat", true); // fileName, saveSynapseInfo
//             }
            
//             if (i == 1078)
//             {
//                 sim.saveSimulation("networkD.dat", true); // fileName, saveSynapseInfo
//             }
            
//             if (i == 1078)
//             {
//                 sim.saveSimulation("networkD.dat", true); // fileName, saveSynapseInfo
//             }
            
//             if (i == 1098)
//             {
//                 sim.saveSimulation("networkF.dat", true); // fileName, saveSynapseInfo
//             }
            
//             if (i == 1198)
//             {
//                 sim.saveSimulation("networkZ.dat", true); // fileName, saveSynapseInfo 
//             }
            
            if (i == 1008)
            {
                sim.saveSimulation("network5p.dat", true); // fileName, saveSynapseInfo
            }
            
            if (i == 1018)
            {
                sim.saveSimulation("network10p.dat", true); // fileName, saveSynapseInfo
            }
            
            if (i == 1028)
            {
                sim.saveSimulation("network15p.dat", true); // fileName, saveSynapseInfo 
            }
            
            if (i == 1038)
            {
                sim.saveSimulation("network20p.dat", true); // fileName, saveSynapseInfo
            }
            
            if (i == 1048)
            {
                sim.saveSimulation("network25p.dat", true); // fileName, saveSynapseInfo
            }
            
            if (i == 1058)
            {
                sim.saveSimulation("network30p.dat", true); // fileName, saveSynapseInfo 
            }
            
            if (i == 1068)
            {
                sim.saveSimulation("network35p.dat", true); // fileName, saveSynapseInfo
            }
            
            if (i == 1078)
            {
                sim.saveSimulation("network40p.dat", true); // fileName, saveSynapseInfo
            }
            
            if (i == 1088)
            {
                sim.saveSimulation("network45p.dat", true); // fileName, saveSynapseInfo 
            }
            
            if (i == 1098)
            {
                sim.saveSimulation("network50p.dat", true); // fileName, saveSynapseInfo
            }
            
            if (i == 1108)
            {
                sim.saveSimulation("network55p.dat", true); // fileName, saveSynapseInfo
            }
            
            if (i == 1118)
            {
                sim.saveSimulation("network60p.dat", true); // fileName, saveSynapseInfo 
            }
            
            if (i == 1128)
            {
                sim.saveSimulation("network65p.dat", true); // fileName, saveSynapseInfo
            }
            
            if (i == 1138)
            {
                sim.saveSimulation("network70p.dat", true); // fileName, saveSynapseInfo 
            }
            
            if (i == 1148)
            {
                sim.saveSimulation("network75p.dat", true); // fileName, saveSynapseInfo
            }
            
            if (i == 1158)
            {
                sim.saveSimulation("network80p.dat", true); // fileName, saveSynapseInfo 
            }
            
            if (i == 1168)
            {
                sim.saveSimulation("network85p.dat", true); // fileName, saveSynapseInfo
            }
            
            if (i == 1178)
            {
                sim.saveSimulation("network90p.dat", true); // fileName, saveSynapseInfo 
            }
            
            if (i == 1188)
            {
                sim.saveSimulation("network95p.dat", true); // fileName, saveSynapseInfo
            }
            
            if (i == 1198)
            {
                sim.saveSimulation("network100p.dat", true); // fileName, saveSynapseInfo 
            }
            
        }
                
    }        
        
    
	// print stopwatch summary
	watch.stop();
    
	return 0;
}