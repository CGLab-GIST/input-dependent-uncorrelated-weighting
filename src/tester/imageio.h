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


#ifndef IMAGEIO_H
#define IMAGEIO_H

#include <string>
#include <vector>

class ImageIO {

public:
	ImageIO(int _numBuffers) {
		numBuffers = _numBuffers;

		if (numBuffers >= 2) {
			subRandImgs.resize(numBuffers);
			subCorrImgs.resize(numBuffers);
		}

		isExistInput = false;
		isExistRef = false;
	}

	void readInputs(std::string &inputPath, std::string &sceneName, std::string &strSpp);
	void readReference(std::string &refPath, std::string &sceneName);

	void saveBuffer(std::string &fileName, std::vector<float> &_out, int nChannel = 3, bool printError = true);
	float calcRelativeL2(std::vector<float> &_img);

public:
	std::vector<float> randImg, corrImg;
	std::vector< std::vector<float> > subRandImgs, subCorrImgs;

	std::vector<float> refImg;
	int numBuffers, xSize, ySize, xSizeRef, ySizeRef;

	bool isExistRef, isExistInput;
};

#endif