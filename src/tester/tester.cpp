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


#include <iostream>
#include <cstring>
#include "imageio.h"
#include "denoiser.h"

#define NUM_BUFFERS		4
#define WIN_SIZE		15


void helpFunction(void) {
	printf("tester.exe [input_path] [ref_path] [scene_name] [method] [spp]\n");
	printf("- input_path    : input path           (e.g., test_scenes/input-crn)\n");
	printf("- ref_path      : reference path       (e.g., test_scenes/reference)\n");
	printf("- scene_name    : scene name           (e.g., bathroom)\n");
	printf("- method        : method               (e.g., crn and L2)\n");
	printf("- spp           : spp                  (e.g., 128)\n");
}

void runDenoiser(std::string& inputPath, std::string& refPath,
	std::string& sceneName, std::string& method, int spp) {
	bool isL2 = false;
	if (method == "L2")
		isL2 = true;

	std::string strSpp = std::to_string(spp);

	ImageIO io(NUM_BUFFERS);
	io.readInputs(inputPath, sceneName, strSpp);
	io.readReference(refPath, sceneName);

	if (!io.isExistInput) {
		return;
	}

	std::vector<float> outImg(3 * io.xSize * io.ySize, 0.f);

	Denoiser denoiser(io.xSize, io.ySize, WIN_SIZE, NUM_BUFFERS);
	denoiser.allocMemory();
	denoiser.runDenoiser(outImg, io.subRandImgs, io.subCorrImgs, spp, isL2);
	std::string outImgName = sceneName + "_" + strSpp + "spp_out";
	io.saveBuffer(outImgName, outImg, 3, true);
	denoiser.deallocMemory();

	printf("[tester.cpp] Done!\n");
}

int main(int argc, char *argv[]) {
	if ((argc != 6) || (strcmp(argv[1], "-h") == 0)) {
		helpFunction();
		return 0;
	}

	std::string inputPath = argv[1];
	std::string refPath = argv[2];
	std::string sceneName = argv[3];
	std::string method = argv[4];
	int spp = std::stoi(argv[5]);

	if (method != "L2" && method != "crn") {
		printf("[tester.cpp] Check the method option! (%s)\n", method.c_str());
		return 0;
	}

	runDenoiser(inputPath, refPath, sceneName, method, spp);

	return 0;
}