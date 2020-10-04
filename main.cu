// CUDA Raytracer written by Nick Dorobek
// May 6 2020


#include "CUDA_RT.h"

#include "Scene.h"
#include "Primitives.h"
#include "Camera.h"
#include "Material.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include "Profiler.h"


__device__ 
VEC3 rayColor(const Ray& r, SceneItem** world, curandState* threadRandState) 
{
	Ray tempRay = r;
	VEC3 currentAtten = VEC3(1.0, 1.0, 1.0);
	VEC3 tempAtten(0,0,0);
	//VEC3 background(.8, .9, 1);
	VEC3 background(0, 0, 0);

	for (int i = 0; i < 50; i++) {
		hitRecord rec;

		// miss
		if (!(*world)->hit(tempRay, 0.001f, FLT_MAX, rec)) {
			currentAtten *= background;
			return currentAtten;
		}

		Ray scattered;
		VEC3 emitted = rec.matPtr->emitted(rec.p);
		// hit an emissive material
		if (!rec.matPtr->scatter(tempRay, rec, tempAtten, scattered, threadRandState)) {
			currentAtten *= emitted;
			return currentAtten;
		}
		tempRay = scattered;

		currentAtten += emitted;
		currentAtten = currentAtten * tempAtten;
	}
	
	return background;
	
}

__global__ void randInit(curandState* rand_state) 
{
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curand_init(1999, 0, 0, rand_state);
	}
}

__global__ 
void renderInit(int maxX, int maxY, curandState* rand_state) 
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= maxX) || (j >= maxY)) return;
	int iPixel = j * maxX + i;
	//Each thread gets same seed, a different sequence number, no offset
	curand_init(1999, iPixel, 0, &rand_state[iPixel]);
}

__global__
void render(float* fb, int maxX, int maxY, int ns, 
	Camera **cam, SceneItem **world, curandState* rand_state)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i >= maxX || j >= maxY) return;
	int iPixel = j * maxX * 3 + i * 3;
	curandState threadRandState = rand_state[j * maxX + i];

	VEC3 col(0.0, 0.0, 0.0);
	for (int s = 0; s < ns; s++) {
		float u = float(i + curand_uniform(&threadRandState)) / float(maxX);
		float v = float(j + curand_uniform(&threadRandState)) / float(maxY);
		Ray r = (*cam)->generateRay(u, v, &threadRandState);
		col += rayColor(r, world, &threadRandState);
	}

	col /= float(ns);
	fb[iPixel + 0] = col[0] <= 0 ? 0 : col[0] >= 1 ? 1 : col[0];
	fb[iPixel + 1] = col[1] <= 0 ? 0 : col[1] >= 1 ? 1 : col[1];
	fb[iPixel + 2] = col[2] <= 0 ? 0 : col[2] >= 1 ? 1 : col[2];
	return;
}



__global__ void createWorld(SceneItem** deviceList, SceneItem** deviceWorld, Camera** deviceCamera, 
	int nx, int ny, curandState* randState) 
{
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curandState threadRandState = *randState;
		deviceList[0] = new Sphere(VEC3(0, -1000.0, -1), 1000,
			new Lambertian(VEC3(0.5, 0.5, 0.5)));
		int i = 1;
		for (int a = -11; a < 11; a++) {
			for (int b = -11; b < 11; b++) {
				float choose_mat = RND_UNF;
				VEC3 center(a + RND_UNF, 0.2, b + RND_UNF);
				if (choose_mat < 0.65f) {
					deviceList[i++] = new Sphere(center, 0.22,
						new Lambertian(VEC3(RND_UNF * RND_UNF, RND_UNF * RND_UNF, RND_UNF * RND_UNF)));
				}
				else if (choose_mat < 0.75f) {
					deviceList[i++] = new Sphere(center, 0.22,
						new Metal(VEC3(0.5f * (1.0f + RND_UNF), 0.5f * (1.0f + RND_UNF), 0.5f * (1.0f + RND_UNF)), 0.5f * RND_UNF));
				}
				else if (choose_mat < 0.9f) {
					deviceList[i++] = new Sphere(center, 0.22, new Dielectric(1.4));
				}
				else {
					deviceList[i++] = new Sphere(center, 0.22, 
						new Hybrid(VEC3(RND_UNF * RND_UNF, RND_UNF * RND_UNF, RND_UNF * RND_UNF), 4));
				}
			}
		}
		deviceList[i++] = new Sphere(VEC3(-2, 10, 2), 4.0, new DiffuseLight(VEC3(5, 4, 3)));
		deviceList[i++] = new Sphere(VEC3(-4, 1, 0), 1.0, new Lambertian(VEC3(0.4, 0.2, 0.5)));
		deviceList[i++] = new Sphere(VEC3(4, 1, 0), 1.0, new Metal(VEC3(0.7, 0.6, 0.5), 0.0));
		*randState = threadRandState;
		*deviceWorld = new Scene(deviceList, 22 * 22 + 1 + 3);

		VEC3 lookFrom(13, 2, 3);
		VEC3 lookAt(0, 0, 0);
		float focusDist = 10.0; (lookFrom - lookAt).norm();
		float aperture = 0.1;
		*deviceCamera = new Camera(lookFrom,
			lookAt,
			VEC3(0, -1, 0),
			30.0,
			float(nx) / float(ny),
			aperture,
			focusDist);
	}
}

__global__ void freeWorld(SceneItem** deviceList, SceneItem** deviceWorld, Camera** deviceCamera) {
	for (int i = 0; i < 22 * 22 + 1 + 3; i++) {
		delete ((Sphere*)deviceList[i])->m_matPtr;
		delete deviceList[i];
	}
	delete* deviceWorld;
	delete* deviceCamera;
}


int main()
{
	const int nx{ 1200 }, ny{ 600 };
	const int nPixels{ nx * ny };
	const int tx{ 8 }, ty{ 4 };
	const int ns{ 16 };

	std::cout << "Rendering an image: " << nx << " x " << ny << " pixels" << std::endl
		<< "Number of samples: " << ns << std::endl
		<< "Tensor Size: " << tx << " x " << ty << std::endl;

	size_t fbSize = 3 * nPixels * sizeof(float);

	// allocate frame buffer
	float* frameBuffer;
	CUDA_CALL(cudaMallocManaged((void**)&frameBuffer, fbSize));

	// allocate random state
	curandState* d_rand_state;
	CUDA_CALL(cudaMalloc((void**)&d_rand_state, nPixels * sizeof(curandState)));
	curandState* d_rand_state2;
	CUDA_CALL(cudaMalloc((void**)&d_rand_state2, 1 * sizeof(curandState)));

	// allocate randomState for world creation
	randInit << <1, 1 >> > (d_rand_state2);
	CUDA_CALL(cudaGetLastError());
	CUDA_CALL(cudaDeviceSynchronize());

	// make our world 
	SceneItem** deviceList;
	int num_hitables = 22 * 22 + 1 + 3;
	CUDA_CALL(cudaMalloc((void**)&deviceList, num_hitables * sizeof(SceneItem*)));
	SceneItem** deviceWorld;
	CUDA_CALL(cudaMalloc((void**)&deviceWorld, sizeof(SceneItem*)));
	Camera** deviceCamera;
	CUDA_CALL(cudaMalloc((void**)&deviceCamera, sizeof(Camera*)));
	createWorld << <1, 1 >> > (deviceList, deviceWorld, deviceCamera, nx, ny, d_rand_state2);
	CUDA_CALL(cudaGetLastError());
	CUDA_CALL(cudaDeviceSynchronize());
	
	// render step
	Profiler::Get().BeginSession("Rendering");
	{
		ProfileTimer timer("Main Render Step");
		dim3 blocks(nx / tx + 1, ny / ty + 1);
		dim3 threads(tx, ty);
		renderInit << <blocks, threads >> > (nx, ny, d_rand_state);
		CUDA_CALL(cudaGetLastError());
		CUDA_CALL(cudaDeviceSynchronize());
		std::cout << "Init'd rendering\n";
		render << <blocks, threads >> > (frameBuffer, nx, ny, ns, deviceCamera, deviceWorld, d_rand_state);
		CUDA_CALL(cudaGetLastError());
		CUDA_CALL(cudaDeviceSynchronize());
	}
	Profiler::Get().EndSession();
	
	std::cout << "Done rendering\n";

	// convert data, and write to file
	unsigned char *fData = new unsigned char[3 * nPixels]();
	for (int i = 0; i < 3 * nPixels; i++) fData[i] = static_cast<unsigned char>(frameBuffer[i] * 255.999f);
	stbi_write_jpg("out.jpg", nx, ny, 3, fData, 100);

	// clean up
	CUDA_CALL(cudaDeviceSynchronize());
	freeWorld<<<1, 1 >>> (deviceList, deviceWorld, deviceCamera);
	CUDA_CALL(cudaGetLastError());
	CUDA_CALL(cudaFree(deviceCamera));
	CUDA_CALL(cudaFree(deviceList));
	CUDA_CALL(cudaFree(deviceWorld));
	CUDA_CALL(cudaFree(d_rand_state));
	CUDA_CALL(cudaFree(frameBuffer));

	cudaDeviceReset();

	return 0;
}
