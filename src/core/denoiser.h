/*
	Copyright (c) 2023 CGLab, GIST. All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, 
	are permitted provided that the following conditions are met:

	- Redistributions of source code must retain the above copyright notice, 
	  this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright notice, 
	  this list of conditions and the following disclaimer in the documentation 
	  and/or other materials provided with the distribution.
	- Neither the name of the copyright holder nor the names of its contributors 
	  may be used to endorse or promote products derived from this software 
	  without specific prior written permission.

	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
	AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
	IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
	ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
	LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
	DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
	SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
	CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
	OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
	OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#ifndef DENOISER_H
#define DENOISER_H

#include <vector>

class Denoiser {

public:
	Denoiser(int _xSize, int _ySize, int _winSize, int _numBuffers) {
		xSize = _xSize;
		ySize = _ySize;
		winSize = _winSize;
		numBuffers = _numBuffers;
	}

	void allocMemory();
	void deallocMemory();

	void runDenoiser(std::vector<float> &outImg,
		std::vector< std::vector<float> > &subImgsY, std::vector< std::vector<float> > &subImgsZ, int spp, bool isL2);

public:
	int xSize, ySize, winSize, numBuffers;

	float *d_outImg, *d_outVar;
	float *d_imgY, *d_imgZ;
	float *d_subImgsY, *d_subImgsZ;
	float *h_subImgsY, *h_subImgsZ;
};

#endif