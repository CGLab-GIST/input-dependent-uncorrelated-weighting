/*
    pbrt source code Copyright(c) 1998-2012 Matt Pharr and Greg Humphreys.

    This file is part of pbrt.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#include "imageio.h"

#include <ImfInputFile.h>
#include <ImfChannelList.h>
#include <ImfFrameBuffer.h>
#include <ImfRgbaFile.h>
#include <half.h>

using namespace Imf;
using namespace Imath;

bool isFileExist(const char *fileName) {
	std::ifstream infile(fileName);
	return infile.good();
}

bool readEXR(const char *name, float *&rgba, int &xRes, int &yRes, int& nChannel) {
	InputFile file(name);
	Box2i dw = file.header().dataWindow();
	xRes = dw.max.x - dw.min.x + 1;
	yRes = dw.max.y - dw.min.y + 1;

	const Imf::ChannelList& clist = file.header().channels();

	nChannel = 0;
	for (Imf::ChannelList::ConstIterator iter = clist.begin(); iter != clist.end(); ++iter) {
		++nChannel;
	}

	float *hrgba = new float[nChannel * xRes * yRes];
	float*hp = hrgba - nChannel * (dw.min.x + dw.min.y * xRes);

	FrameBuffer frameBuffer;
	int i = 0;
	for (Imf::ChannelList::ConstIterator iter = clist.begin(); iter != clist.end(); ++iter) {
		frameBuffer.insert(iter.name(), Slice(FLOAT, (char *)hp + i * sizeof(float), nChannel * sizeof(float), xRes * nChannel * sizeof(float), 1, 1, 0.0));
		i++;
	}

	file.setFrameBuffer(frameBuffer);
	file.readPixels(dw.min.y, dw.max.y);

	rgba = new float[nChannel * xRes * yRes];
	for (int i = 0; i < nChannel * xRes * yRes; ++i)
		rgba[i] = hrgba[i];
	delete[] hrgba;

	return true;
}

void writeEXR(const char *name, const std::vector<float>& pixels, int xRes, int yRes, int nChannel) {
	Rgba *hrgba = new Rgba[xRes * yRes];
	for (int i = 0; i < xRes * yRes; ++i) {
		if (nChannel == 4)
			hrgba[i] = Rgba(pixels[4 * i], pixels[4 * i + 1], pixels[4 * i + 2], 1.);
		else if (nChannel == 3)
			hrgba[i] = Rgba(pixels[3 * i], pixels[3 * i + 1], pixels[3 * i + 2], 1.);
		else if (nChannel == 1)
			hrgba[i] = Rgba(pixels[i], pixels[i], pixels[i], 1.);
	}

	Box2i displayWindow(V2i(0, 0), V2i(xRes - 1, yRes - 1));
	Box2i dataWindow = displayWindow;

	RgbaOutputFile file(name, displayWindow, dataWindow, WRITE_RGBA);
	file.setFrameBuffer(hrgba, 1, xRes);
	try {
		file.writePixels(yRes);
	}
	catch (const std::exception &e) {
		fprintf(stderr, "Unable to write image file \"%s\": %s", name, e.what());
	}

	delete[] hrgba;
}

bool fillBuffer(std::vector<float>& buf, const char* strFile, int& xRes, int& yRes) {
	float *rgba;
	int nChannel;
	bool isBGR = true;

	if (!readEXR(strFile, rgba, xRes, yRes, nChannel))
		return false;

	buf.resize(xRes * yRes * 3);
	if ((nChannel != 1) && (nChannel != 3) && (nChannel != 4)) {
		printf("[imageio.cpp] 1, 3 or 4 channels are only supported.\n");
		return false;
	}
	if (nChannel == 1) {
		for (int i = 0; i < xRes * yRes; ++i) {
			buf[i] = rgba[i];
		}
	}
	else {
		for (int i = 0; i < xRes * yRes; ++i) {
			if (isBGR) {
				int c = nChannel - 1;
				buf[3 * i + 0] = rgba[i * nChannel + c--];
				buf[3 * i + 1] = rgba[i * nChannel + c--];
				buf[3 * i + 2] = rgba[i * nChannel + c--];
			}
			else {
				int c = 0;
				buf[3 * i + 0] = rgba[i * nChannel + c++];
				buf[3 * i + 1] = rgba[i * nChannel + c++];
				buf[3 * i + 2] = rgba[i * nChannel + c++];
			}
		}
	}
	delete[] rgba;

	return true;
}

bool readEXR(std::vector<float>& _buf, int& xSize, int& ySize, const std::string fileName) {
	if (isFileExist(fileName.c_str())) {
		fillBuffer(_buf, fileName.c_str(), xSize, ySize);
		return true;
	}
	else {
		printf("[imageio.cpp] %s doesn't exist\n", fileName.c_str());
		return false;
	}
}

void ImageIO::readInputs(std::string &inputPath, std::string &sceneName, std::string &strSpp) {
	std::string imgPrefix = inputPath + "/" + sceneName + "/" + sceneName + "_" + strSpp + "spp_";
	isExistInput = readEXR(randImg, xSize, ySize, imgPrefix + "randImg.exr");
	isExistInput = readEXR(corrImg, xSize, ySize, imgPrefix + "corrImg.exr");
	for (int bufferIdx = 0; bufferIdx < numBuffers; bufferIdx++) {
		std::string bufferSuffix = std::to_string(bufferIdx);
		isExistInput = readEXR(subRandImgs[bufferIdx], xSize, ySize, imgPrefix + "randImg" + bufferSuffix + ".exr");
		isExistInput = readEXR(subCorrImgs[bufferIdx], xSize, ySize, imgPrefix + "corrImg" + bufferSuffix + ".exr");
	}

	printf("[imageio.cpp] Inputs have been loaded!\n");
}

void ImageIO::readReference(std::string &refPath, std::string &sceneName) {
	std::string imgPrefix = refPath + "/" + sceneName + "/" + sceneName;
	isExistRef = readEXR(refImg, xSizeRef, ySizeRef, imgPrefix + "_ref.exr");
}

void ImageIO::saveBuffer(std::string &fileName, std::vector<float> &img, int nChannel, bool printError) {
	if (isExistRef && printError) {
		float cRelL2 = calcRelativeL2(img);
		printf("[imageio.cpp] RelL2 of output : %.6f\n", cRelL2);

		std::stringstream _stream;
		_stream.precision(6);
		_stream << std::fixed;

		_stream << cRelL2;
		writeEXR((fileName + "_" + _stream.str() + ".exr").c_str(), img, xSize, ySize, nChannel);
	}
	else {
		writeEXR((fileName + ".exr").c_str(), img, xSize, ySize, nChannel);
	}
}

float ImageIO::calcRelativeL2(std::vector<float> &_img) {
	float accVal = 0.f;
	for (int pixIdx = 0; pixIdx < xSize * ySize; pixIdx++) {
		float numerator = 0.f, denominator = 0.f;
		for (int ch = 0; ch < 3; ch++) {
			float diff = (_img[3 * pixIdx + ch] - refImg[3 * pixIdx + ch]);
			numerator += diff * diff;
			denominator += refImg[3 * pixIdx + ch] / 3.f;
		}
		accVal += numerator / (denominator * denominator + 1e-2f);
	}
	float relL2 = accVal / (3.f * xSize * ySize);
	return relL2;
}