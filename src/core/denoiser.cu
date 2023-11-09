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

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <vector_types.h>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include "denoiser.h"

#define IMAD(a, b, c)			( __mul24((a), (b)) + (c) )

inline int iDivUp(int a, int b) {
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

__forceinline__ __host__ __device__ void operator+=(float4 &a, float4 b) {
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
}

__forceinline__ __host__ __device__ float4 operator*(float b, float4 a) {
	return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}

__forceinline__ __host__ __device__ float4 operator*(float4 a, float4 b) {
	return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

__forceinline__ __host__ __device__ float4 operator+(float4 a, float4 b) {
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__forceinline__ __host__ __device__ float4 operator-(float4 a, float4 b) {
	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}


__global__ void KernelCalcDenoisedVariance(float *outVar, 
	float *subImgsY, float *subImgsZ, int xSize, int ySize, int winSize, int spp, float paramGamma) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= xSize || cy >= ySize)
		return;

	const int cIdx = cy * xSize + cx;
	const int halfWinSize = winSize / 2;
	const int winSizeSqr = winSize * winSize;
	const int colorDim = 3;
	const int nPix = colorDim * xSize * ySize;

	const int bufferIdxA0 = 0;
	const int bufferIdxA1 = 2;
	const int bufferIdxB0 = 1;
	const int bufferIdxB1 = 3;

	int cIdxA0 = bufferIdxA0 * nPix + cIdx * colorDim;
	int cIdxA1 = bufferIdxA1 * nPix + cIdx * colorDim;
	int cIdxB0 = bufferIdxB0 * nPix + cIdx * colorDim;
	int cIdxB1 = bufferIdxB1 * nPix + cIdx * colorDim;

	const float4& cImgYA0 = make_float4(subImgsY[cIdxA0 + 0], subImgsY[cIdxA0 + 1], subImgsY[cIdxA0 + 2], 0.f);
	const float4& cImgYA1 = make_float4(subImgsY[cIdxA1 + 0], subImgsY[cIdxA1 + 1], subImgsY[cIdxA1 + 2], 0.f);
	const float4& cImgYB0 = make_float4(subImgsY[cIdxB0 + 0], subImgsY[cIdxB0 + 1], subImgsY[cIdxB0 + 2], 0.f);
	const float4& cImgYB1 = make_float4(subImgsY[cIdxB1 + 0], subImgsY[cIdxB1 + 1], subImgsY[cIdxB1 + 2], 0.f);
	const float4& cImgZA0 = make_float4(subImgsZ[cIdxA0 + 0], subImgsZ[cIdxA0 + 1], subImgsZ[cIdxA0 + 2], 0.f);
	const float4& cImgZA1 = make_float4(subImgsZ[cIdxA1 + 0], subImgsZ[cIdxA1 + 1], subImgsZ[cIdxA1 + 2], 0.f);
	const float4& cImgZB0 = make_float4(subImgsZ[cIdxB0 + 0], subImgsZ[cIdxB0 + 1], subImgsZ[cIdxB0 + 2], 0.f);
	const float4& cImgZB1 = make_float4(subImgsZ[cIdxB1 + 0], subImgsZ[cIdxB1 + 1], subImgsZ[cIdxB1 + 2], 0.f);
	
	const float4 cImgYA = 0.5f * (cImgYA0 + cImgYA1);
	const float4 cImgYB = 0.5f * (cImgYB0 + cImgYB1);
	const float4 cImgZA = 0.5f * (cImgZA0 + cImgZA1);
	const float4 cImgZB = 0.5f * (cImgZB0 + cImgZB1);

	float4 accColA = make_float4(0.f, 0.f, 0.f, 0.f);
	float4 accColB = make_float4(0.f, 0.f, 0.f, 0.f);
	float4 accImgY = 0.5f * (cImgYA + cImgYB);
	for (int iy = cy - halfWinSize, winIdx = 0; iy <= cy + halfWinSize; iy++) {
		for (int ix = cx - halfWinSize; ix <= cx + halfWinSize; ix++, winIdx++) {
			int x = (ix >= xSize) ? 2 * xSize - 2 - ix : abs(ix);
			int y = (iy >= ySize) ? 2 * ySize - 2 - iy : abs(iy);
			int iIdx = y * xSize + x;

			if (ix == cx && iy == cy)
				continue;

			int iIdxA0 = bufferIdxA0 * nPix + iIdx * colorDim;
			int iIdxA1 = bufferIdxA1 * nPix + iIdx * colorDim;
			int iIdxB0 = bufferIdxB0 * nPix + iIdx * colorDim;
			int iIdxB1 = bufferIdxB1 * nPix + iIdx * colorDim;

			const float4& iImgYA0 = make_float4(subImgsY[iIdxA0 + 0], subImgsY[iIdxA0 + 1], subImgsY[iIdxA0 + 2], 0.f);
			const float4& iImgYA1 = make_float4(subImgsY[iIdxA1 + 0], subImgsY[iIdxA1 + 1], subImgsY[iIdxA1 + 2], 0.f);
			const float4& iImgYB0 = make_float4(subImgsY[iIdxB0 + 0], subImgsY[iIdxB0 + 1], subImgsY[iIdxB0 + 2], 0.f);
			const float4& iImgYB1 = make_float4(subImgsY[iIdxB1 + 0], subImgsY[iIdxB1 + 1], subImgsY[iIdxB1 + 2], 0.f);
			const float4& iImgZA0 = make_float4(subImgsZ[iIdxA0 + 0], subImgsZ[iIdxA0 + 1], subImgsZ[iIdxA0 + 2], 0.f);
			const float4& iImgZA1 = make_float4(subImgsZ[iIdxA1 + 0], subImgsZ[iIdxA1 + 1], subImgsZ[iIdxA1 + 2], 0.f);
			const float4& iImgZB0 = make_float4(subImgsZ[iIdxB0 + 0], subImgsZ[iIdxB0 + 1], subImgsZ[iIdxB0 + 2], 0.f);
			const float4& iImgZB1 = make_float4(subImgsZ[iIdxB1 + 0], subImgsZ[iIdxB1 + 1], subImgsZ[iIdxB1 + 2], 0.f);
			
			const float4 iImgYA = 0.5f * (iImgYA0 + iImgYA1);
			const float4 iImgYB = 0.5f * (iImgYB0 + iImgYB1);
			const float4 iImgZA = 0.5f * (iImgZA0 + iImgZA1);
			const float4 iImgZB = 0.5f * (iImgZB0 + iImgZB1);

			// Calculate a simple variance-based weighting (Eq. 8)
			float4 zDiffA0 = cImgZA0 - iImgZA0;
			float4 zDiffA1 = cImgZA1 - iImgZA1;
			float4 zDiffB0 = cImgZB0 - iImgZB0;
			float4 zDiffB1 = cImgZB1 - iImgZB1;
			float4 zVarA = make_float4((zDiffA0.x - zDiffA1.x) * (zDiffA0.x - zDiffA1.x), 
									   (zDiffA0.y - zDiffA1.y) * (zDiffA0.y - zDiffA1.y), 
									   (zDiffA0.z - zDiffA1.z) * (zDiffA0.z - zDiffA1.z), 0.f);
			float4 zVarB = make_float4((zDiffB0.x - zDiffB1.x) * (zDiffB0.x - zDiffB1.x), 
									   (zDiffB0.y - zDiffB1.y) * (zDiffB0.y - zDiffB1.y), 
									   (zDiffB0.z - zDiffB1.z) * (zDiffB0.z - zDiffB1.z), 0.f);
			float4 cWgtA = make_float4(expf(-paramGamma * (float)spp * zVarA.x),
									   expf(-paramGamma * (float)spp * zVarA.y),
									   expf(-paramGamma * (float)spp * zVarA.z), 0.f);
			float4 cWgtB = make_float4(expf(-paramGamma * (float)spp * zVarB.x),
									   expf(-paramGamma * (float)spp * zVarB.y),
									   expf(-paramGamma * (float)spp * zVarB.z), 0.f);
			
			accColA += cWgtA * ((cImgZA - iImgZA) - (cImgYA - iImgYA));
			accColB += cWgtB * ((cImgZB - iImgZB) - (cImgYB - iImgYB));
			accImgY += 0.5f * (iImgYA + iImgYB);
		}
	}

	float invEle = 1.f / ((float)winSizeSqr - 1.f);
	float4 outColA = make_float4(cImgYA.x + invEle * accColA.x,
								 cImgYA.y + invEle * accColA.y,
								 cImgYA.z + invEle * accColA.z, 0.f);
	float4 outColB = make_float4(cImgYB.x + invEle * accColB.x,
								 cImgYB.y + invEle * accColB.y,
								 cImgYB.z + invEle * accColB.z, 0.f);

	float4 avgImgY = (1.f / (float)winSizeSqr) * accImgY;
	float4 varNumerator = (outColA - outColB) * (outColA - outColB);
	float4 varDenominator = avgImgY * avgImgY;
	float avgDenoisedVar = (varNumerator.x / (varDenominator.x + 1e-2f)
						  + varNumerator.y / (varDenominator.y + 1e-2f)
						  + varNumerator.z / (varDenominator.z + 1e-2f)) / 3.f;

	outVar[cIdx] = avgDenoisedVar;
}

__global__ void KernelDenoising(float *outImg, float *subImgsY, float *subImgsZ,
	int xSize, int ySize, int winSize, int spp, float paramGamma) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= xSize || cy >= ySize)
		return;

	const int cIdx = cy * xSize + cx;
	const int halfWinSize = winSize / 2;
	const int winSizeSqr = winSize * winSize;
	const int colorDim = 3;
	const int nPix = colorDim * xSize * ySize;

	const int bufferIdxA0 = 0;
	const int bufferIdxA1 = 2;
	const int bufferIdxB0 = 1;
	const int bufferIdxB1 = 3;

	int cIdxA0 = bufferIdxA0 * nPix + cIdx * colorDim;
	int cIdxA1 = bufferIdxA1 * nPix + cIdx * colorDim;
	int cIdxB0 = bufferIdxB0 * nPix + cIdx * colorDim;
	int cIdxB1 = bufferIdxB1 * nPix + cIdx * colorDim;

	const float4& cImgYA0 = make_float4(subImgsY[cIdxA0 + 0], subImgsY[cIdxA0 + 1], subImgsY[cIdxA0 + 2], 0.f);
	const float4& cImgYA1 = make_float4(subImgsY[cIdxA1 + 0], subImgsY[cIdxA1 + 1], subImgsY[cIdxA1 + 2], 0.f);
	const float4& cImgYB0 = make_float4(subImgsY[cIdxB0 + 0], subImgsY[cIdxB0 + 1], subImgsY[cIdxB0 + 2], 0.f);
	const float4& cImgYB1 = make_float4(subImgsY[cIdxB1 + 0], subImgsY[cIdxB1 + 1], subImgsY[cIdxB1 + 2], 0.f);
	const float4& cImgZA0 = make_float4(subImgsZ[cIdxA0 + 0], subImgsZ[cIdxA0 + 1], subImgsZ[cIdxA0 + 2], 0.f);
	const float4& cImgZA1 = make_float4(subImgsZ[cIdxA1 + 0], subImgsZ[cIdxA1 + 1], subImgsZ[cIdxA1 + 2], 0.f);
	const float4& cImgZB0 = make_float4(subImgsZ[cIdxB0 + 0], subImgsZ[cIdxB0 + 1], subImgsZ[cIdxB0 + 2], 0.f);
	const float4& cImgZB1 = make_float4(subImgsZ[cIdxB1 + 0], subImgsZ[cIdxB1 + 1], subImgsZ[cIdxB1 + 2], 0.f);
	
	const float4 cImgYA = 0.5f * (cImgYA0 + cImgYA1);
	const float4 cImgYB = 0.5f * (cImgYB0 + cImgYB1);
	const float4 cImgZA = 0.5f * (cImgZA0 + cImgZA1);
	const float4 cImgZB = 0.5f * (cImgZB0 + cImgZB1);
	const float4 cImgY = 0.5f * (cImgYA + cImgYB);
	const float4 cImgZ = 0.5f * (cImgZA + cImgZB);

	float4 accCol = make_float4(0.f, 0.f, 0.f, 0.f);
	for (int iy = cy - halfWinSize, winIdx = 0; iy <= cy + halfWinSize; iy++) {
		for (int ix = cx - halfWinSize; ix <= cx + halfWinSize; ix++, winIdx++) {
			int x = (ix >= xSize) ? 2 * xSize - 2 - ix : abs(ix);
			int y = (iy >= ySize) ? 2 * ySize - 2 - iy : abs(iy);
			int iIdx = y * xSize + x;

			if (ix == cx && iy == cy)
				continue;

			int iIdxA0 = bufferIdxA0 * nPix + iIdx * colorDim;
			int iIdxA1 = bufferIdxA1 * nPix + iIdx * colorDim;
			int iIdxB0 = bufferIdxB0 * nPix + iIdx * colorDim;
			int iIdxB1 = bufferIdxB1 * nPix + iIdx * colorDim;

			const float4& iImgYA0 = make_float4(subImgsY[iIdxA0 + 0], subImgsY[iIdxA0 + 1], subImgsY[iIdxA0 + 2], 0.f);
			const float4& iImgYA1 = make_float4(subImgsY[iIdxA1 + 0], subImgsY[iIdxA1 + 1], subImgsY[iIdxA1 + 2], 0.f);
			const float4& iImgYB0 = make_float4(subImgsY[iIdxB0 + 0], subImgsY[iIdxB0 + 1], subImgsY[iIdxB0 + 2], 0.f);
			const float4& iImgYB1 = make_float4(subImgsY[iIdxB1 + 0], subImgsY[iIdxB1 + 1], subImgsY[iIdxB1 + 2], 0.f);
			const float4& iImgZA0 = make_float4(subImgsZ[iIdxA0 + 0], subImgsZ[iIdxA0 + 1], subImgsZ[iIdxA0 + 2], 0.f);
			const float4& iImgZA1 = make_float4(subImgsZ[iIdxA1 + 0], subImgsZ[iIdxA1 + 1], subImgsZ[iIdxA1 + 2], 0.f);
			const float4& iImgZB0 = make_float4(subImgsZ[iIdxB0 + 0], subImgsZ[iIdxB0 + 1], subImgsZ[iIdxB0 + 2], 0.f);
			const float4& iImgZB1 = make_float4(subImgsZ[iIdxB1 + 0], subImgsZ[iIdxB1 + 1], subImgsZ[iIdxB1 + 2], 0.f);
			
			const float4 iImgYA = 0.5f * (iImgYA0 + iImgYA1);
			const float4 iImgYB = 0.5f * (iImgYB0 + iImgYB1);
			const float4 iImgZA = 0.5f * (iImgZA0 + iImgZA1);
			const float4 iImgZB = 0.5f * (iImgZB0 + iImgZB1);
			const float4 iImgY = 0.5f * (iImgYA + iImgYB);
			const float4 iImgZ = 0.5f * (iImgZA + iImgZB);

			// Calculate a simple variance-based weighting (Eq. 8)
			float4 zDiffA = cImgZA - iImgZA;
			float4 zDiffB = cImgZB - iImgZB;
			float4 zVar = make_float4((zDiffA.x - zDiffB.x) * (zDiffA.x - zDiffB.x),
									  (zDiffA.y - zDiffB.y) * (zDiffA.y - zDiffB.y),
									  (zDiffA.z - zDiffB.z) * (zDiffA.z - zDiffB.z), 0.f);
			float4 cWgt = make_float4(expf(-paramGamma * (float)spp * zVar.x),
									  expf(-paramGamma * (float)spp * zVar.y),
									  expf(-paramGamma * (float)spp * zVar.z), 0.f);
			
			accCol += cWgt * ((cImgZ - iImgZ) - (cImgY - iImgY));
		}
	}

	float invEle = 1.f / ((float)winSizeSqr - 1.f);
	outImg[colorDim * cIdx + 0] = cImgY.x + invEle * accCol.x;
	outImg[colorDim * cIdx + 1] = cImgY.y + invEle * accCol.y;
	outImg[colorDim * cIdx + 2] = cImgY.z + invEle * accCol.z;
}

__global__ void KernelDenoisingForL2(float *outImg, float *subImgsY, float *subImgsZ,
	int xSize, int ySize, int winSize, int spp, float paramGamma) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= xSize || cy >= ySize)
		return;

	const int cIdx = cy * xSize + cx;
	const int halfWinSize = winSize / 2;
	const int winSizeSqr = winSize * winSize;
	const int colorDim = 3;
	const int nPix = colorDim * xSize * ySize;

	const int bufferIdxA0 = 0;
	const int bufferIdxA1 = 2;
	const int bufferIdxB0 = 1;
	const int bufferIdxB1 = 3;

	int cIdxA0 = bufferIdxA0 * nPix + cIdx * colorDim;
	int cIdxA1 = bufferIdxA1 * nPix + cIdx * colorDim;
	int cIdxB0 = bufferIdxB0 * nPix + cIdx * colorDim;
	int cIdxB1 = bufferIdxB1 * nPix + cIdx * colorDim;

	const float4& cImgYA0 = make_float4(subImgsY[cIdxA0 + 0], subImgsY[cIdxA0 + 1], subImgsY[cIdxA0 + 2], 0.f);
	const float4& cImgYA1 = make_float4(subImgsY[cIdxA1 + 0], subImgsY[cIdxA1 + 1], subImgsY[cIdxA1 + 2], 0.f);
	const float4& cImgYB0 = make_float4(subImgsY[cIdxB0 + 0], subImgsY[cIdxB0 + 1], subImgsY[cIdxB0 + 2], 0.f);
	const float4& cImgYB1 = make_float4(subImgsY[cIdxB1 + 0], subImgsY[cIdxB1 + 1], subImgsY[cIdxB1 + 2], 0.f);
	const float4& cImgZA0 = make_float4(subImgsZ[cIdxA0 + 0], subImgsZ[cIdxA0 + 1], subImgsZ[cIdxA0 + 2], 0.f);
	const float4& cImgZA1 = make_float4(subImgsZ[cIdxA1 + 0], subImgsZ[cIdxA1 + 1], subImgsZ[cIdxA1 + 2], 0.f);
	const float4& cImgZB0 = make_float4(subImgsZ[cIdxB0 + 0], subImgsZ[cIdxB0 + 1], subImgsZ[cIdxB0 + 2], 0.f);
	const float4& cImgZB1 = make_float4(subImgsZ[cIdxB1 + 0], subImgsZ[cIdxB1 + 1], subImgsZ[cIdxB1 + 2], 0.f);
	
	const float4 cImgYA = 0.5f * (cImgYA0 + cImgYA1);
	const float4 cImgYB = 0.5f * (cImgYB0 + cImgYB1);
	const float4 cImgZA = 0.5f * (cImgZA0 + cImgZA1);
	const float4 cImgZB = 0.5f * (cImgZB0 + cImgZB1);

	float4 accColA = make_float4(0.f, 0.f, 0.f, 0.f);
	float4 accColB = make_float4(0.f, 0.f, 0.f, 0.f);
	for (int iy = cy - halfWinSize, winIdx = 0; iy <= cy + halfWinSize; iy++) {
		for (int ix = cx - halfWinSize; ix <= cx + halfWinSize; ix++, winIdx++) {
			int x = (ix >= xSize) ? 2 * xSize - 2 - ix : abs(ix);
			int y = (iy >= ySize) ? 2 * ySize - 2 - iy : abs(iy);
			int iIdx = y * xSize + x;

			if (ix == cx && iy == cy)
				continue;

			int iIdxA0 = bufferIdxA0 * nPix + iIdx * colorDim;
			int iIdxA1 = bufferIdxA1 * nPix + iIdx * colorDim;
			int iIdxB0 = bufferIdxB0 * nPix + iIdx * colorDim;
			int iIdxB1 = bufferIdxB1 * nPix + iIdx * colorDim;

			const float4& iImgYA0 = make_float4(subImgsY[iIdxA0 + 0], subImgsY[iIdxA0 + 1], subImgsY[iIdxA0 + 2], 0.f);
			const float4& iImgYA1 = make_float4(subImgsY[iIdxA1 + 0], subImgsY[iIdxA1 + 1], subImgsY[iIdxA1 + 2], 0.f);
			const float4& iImgYB0 = make_float4(subImgsY[iIdxB0 + 0], subImgsY[iIdxB0 + 1], subImgsY[iIdxB0 + 2], 0.f);
			const float4& iImgYB1 = make_float4(subImgsY[iIdxB1 + 0], subImgsY[iIdxB1 + 1], subImgsY[iIdxB1 + 2], 0.f);
			const float4& iImgZA0 = make_float4(subImgsZ[iIdxA0 + 0], subImgsZ[iIdxA0 + 1], subImgsZ[iIdxA0 + 2], 0.f);
			const float4& iImgZA1 = make_float4(subImgsZ[iIdxA1 + 0], subImgsZ[iIdxA1 + 1], subImgsZ[iIdxA1 + 2], 0.f);
			const float4& iImgZB0 = make_float4(subImgsZ[iIdxB0 + 0], subImgsZ[iIdxB0 + 1], subImgsZ[iIdxB0 + 2], 0.f);
			const float4& iImgZB1 = make_float4(subImgsZ[iIdxB1 + 0], subImgsZ[iIdxB1 + 1], subImgsZ[iIdxB1 + 2], 0.f);
			
			const float4 iImgYA = 0.5f * (iImgYA0 + iImgYA1);
			const float4 iImgYB = 0.5f * (iImgYB0 + iImgYB1);
			const float4 iImgZA = 0.5f * (iImgZA0 + iImgZA1);
			const float4 iImgZB = 0.5f * (iImgZB0 + iImgZB1);

			// Calculate a simple variance-based weighting (Eq. 8)
			float4 zDiffA0 = cImgZA0 - iImgZA0;
			float4 zDiffA1 = cImgZA1 - iImgZA1;
			float4 zDiffB0 = cImgZB0 - iImgZB0;
			float4 zDiffB1 = cImgZB1 - iImgZB1;
			float4 zVarA = make_float4((zDiffA0.x - zDiffA1.x) * (zDiffA0.x - zDiffA1.x), 
									   (zDiffA0.y - zDiffA1.y) * (zDiffA0.y - zDiffA1.y), 
									   (zDiffA0.z - zDiffA1.z) * (zDiffA0.z - zDiffA1.z), 0.f);
			float4 zVarB = make_float4((zDiffB0.x - zDiffB1.x) * (zDiffB0.x - zDiffB1.x), 
									   (zDiffB0.y - zDiffB1.y) * (zDiffB0.y - zDiffB1.y), 
									   (zDiffB0.z - zDiffB1.z) * (zDiffB0.z - zDiffB1.z), 0.f);
			float4 cWgtA = make_float4(expf(-paramGamma  * (float)spp * zVarA.x),
									   expf(-paramGamma  * (float)spp * zVarA.y),
									   expf(-paramGamma  * (float)spp * zVarA.z), 0.f);
			float4 cWgtB = make_float4(expf(-paramGamma  * (float)spp * zVarB.x),
									   expf(-paramGamma  * (float)spp * zVarB.y),
									   expf(-paramGamma  * (float)spp * zVarB.z), 0.f);
			
			accColA += cWgtA * ((cImgZA - iImgZA) - (cImgYA - iImgYA));
			accColB += cWgtB * ((cImgZB - iImgZB) - (cImgYB - iImgYB));
		}
	}

	float invEle = 1.f / ((float)winSizeSqr - 1.f);
	float4 outColA = make_float4(cImgYA.x + invEle * accColA.x,
								 cImgYA.y + invEle * accColA.y,
								 cImgYA.z + invEle * accColA.z, 0.f);
	float4 outColB = make_float4(cImgYB.x + invEle * accColB.x,
								 cImgYB.y + invEle * accColB.y,
								 cImgYB.z + invEle * accColB.z, 0.f);

	outImg[colorDim * cIdx + 0] = 0.5f * (outColA.x + outColB.x);
	outImg[colorDim * cIdx + 1] = 0.5f * (outColA.y + outColB.y);
	outImg[colorDim * cIdx + 2] = 0.5f * (outColA.z + outColB.z);
}


float applyParameterSelection(dim3 threads, dim3 grid, float *d_outVar, float *d_subImgsY, float *d_subImgsZ,
	int xSize, int ySize, int winSize, int spp) {
	const int maxIter = 10;
	const float paramGammaSet[maxIter] = { 0.01f, 0.025f, 0.05f, 0.1f, 0.2f,
										   0.5f, 1.f, 1.5f, 2.f, 2.5f };

	int minIterIdx = -1;
	float minDenoisedVar = FLT_MAX;
	for (int iter = 0; iter < maxIter; iter++) {
		float cParamGamma = paramGammaSet[iter];

		KernelCalcDenoisedVariance << < grid, threads >> >(d_outVar, d_subImgsY, d_subImgsZ,
			xSize, ySize, winSize, spp, cParamGamma);
		cudaDeviceSynchronize();

		thrust::device_ptr<float> d_outVar_ptr(d_outVar);
		float accDenoisedVar = thrust::reduce(d_outVar_ptr, d_outVar_ptr + xSize * ySize);
		float avgDenoisedVar = accDenoisedVar / (float)(xSize * ySize);

		if (avgDenoisedVar < minDenoisedVar) {
			minDenoisedVar = avgDenoisedVar;
			minIterIdx = iter;
		}
	}

	if (minIterIdx < 0)
		printf("[denoiser.cu] Please check minIterIdx!\n");

	return paramGammaSet[minIterIdx];
}

void Denoiser::allocMemory() {
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount > 1) {
		printf("[denoiser.cu] Multiple GPUs exist, which is not supported!\n");
		cudaSetDevice(0);
	}

	cudaMalloc((void **)&d_outImg, 3 * xSize * ySize * sizeof(float));
	cudaMalloc((void **)&d_outVar, xSize * ySize * sizeof(float));

	cudaMalloc((void **)&d_imgY, 3 * xSize * ySize * sizeof(float));
	cudaMalloc((void **)&d_imgZ, 3 * xSize * ySize * sizeof(float));
	cudaMalloc((void **)&d_subImgsY, numBuffers * 3 * xSize * ySize * sizeof(float));
	cudaMalloc((void **)&d_subImgsZ, numBuffers * 3 * xSize * ySize * sizeof(float));

	h_subImgsY = new float[numBuffers * 3 * xSize * ySize];
	h_subImgsZ = new float[numBuffers * 3 * xSize * ySize];
}

void Denoiser::deallocMemory() {
	cudaFree(d_outImg);
	cudaFree(d_outVar);

	cudaFree(d_imgY);
	cudaFree(d_imgZ);
	cudaFree(d_subImgsY);
	cudaFree(d_subImgsZ);

	delete[] h_subImgsY;
	delete[] h_subImgsZ;
}

void Denoiser::runDenoiser(std::vector<float> &outImg,
	std::vector< std::vector<float> > &subImgsY, std::vector< std::vector<float> > &subImgsZ, int spp, bool isL2) {
	const int blockDim = 16;
	dim3 threads(blockDim, blockDim);
	dim3 grid(iDivUp(xSize, blockDim), iDivUp(ySize, blockDim));

	const int nPix = 3 * xSize * ySize;
	int nSppForCorrEst = spp / 2;
	int nSppForSubBuffer = nSppForCorrEst / numBuffers;
	
	if (isL2) {
		nSppForCorrEst = spp;
		nSppForSubBuffer = nSppForCorrEst / numBuffers;

		// JH: decorrelation
		const int bufferIdxA0 = 0, bufferIdxA1 = 2;
		const int bufferIdxB0 = 1, bufferIdxB1 = 3;
		for (int pixIdx = 0; pixIdx < nPix; pixIdx++) {
			h_subImgsY[bufferIdxA0 * nPix + pixIdx] = subImgsY[bufferIdxA0][pixIdx];
			h_subImgsY[bufferIdxA1 * nPix + pixIdx] = subImgsY[bufferIdxA1][pixIdx];
			h_subImgsY[bufferIdxB0 * nPix + pixIdx] = subImgsY[bufferIdxB0][pixIdx];
			h_subImgsY[bufferIdxB1 * nPix + pixIdx] = subImgsY[bufferIdxB1][pixIdx];

			h_subImgsZ[bufferIdxA0 * nPix + pixIdx] = subImgsZ[bufferIdxB0][pixIdx];
			h_subImgsZ[bufferIdxA1 * nPix + pixIdx] = subImgsZ[bufferIdxB1][pixIdx];
			h_subImgsZ[bufferIdxB0 * nPix + pixIdx] = subImgsZ[bufferIdxA0][pixIdx];
			h_subImgsZ[bufferIdxB1 * nPix + pixIdx] = subImgsZ[bufferIdxA1][pixIdx];
		}
	}
	else {
		for (int bufferIdx = 0; bufferIdx < numBuffers; bufferIdx++) {
			for (int pixIdx = 0; pixIdx < nPix; pixIdx++) {
				h_subImgsY[bufferIdx * nPix + pixIdx] = subImgsY[bufferIdx][pixIdx];
				h_subImgsZ[bufferIdx * nPix + pixIdx] = subImgsZ[bufferIdx][pixIdx];
			}
		}
	}

	cudaMemcpy(d_subImgsY, h_subImgsY, numBuffers * nPix * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_subImgsZ, h_subImgsZ, numBuffers * nPix * sizeof(float), cudaMemcpyHostToDevice);

	printf("[denoiser.cu] Denoising starts!\n");
	// ==============================================================
	// Parameter selection (last paragraph in Sec. 4.2)
	float optParamGamma = applyParameterSelection(threads, grid, d_outVar, d_subImgsY, d_subImgsZ,
		xSize, ySize, winSize, nSppForSubBuffer);
	
	// Denoising with an example kernel satisfying our conditions
	// - Simple variance-based weighting with B=2 (Eq. 8 in Sec. 4.2)
	if (isL2) {
		KernelDenoisingForL2 << <grid, threads >> >(d_outImg, d_subImgsY, d_subImgsZ,
			xSize, ySize, winSize, nSppForSubBuffer, optParamGamma);
		cudaDeviceSynchronize();
	}
	else {
		int nSppForHalfBuffer = nSppForCorrEst / 2;
		KernelDenoising << <grid, threads >> >(d_outImg, d_subImgsY, d_subImgsZ,
			xSize, ySize, winSize, nSppForHalfBuffer, optParamGamma);
		cudaDeviceSynchronize();
	}
	// ==============================================================
	printf("[denoiser.cu] Denoising done!\n");

	// Final output saving
	cudaMemcpy(outImg.data(), d_outImg, nPix * sizeof(float), cudaMemcpyDeviceToHost);
	cudaGetLastError();
}