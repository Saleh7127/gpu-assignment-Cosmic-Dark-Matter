# GPU Assignment - Cosmic Dark Matter

This project implements a GPU-accelerated application for the calculation of histograms related to cosmic dark matter data. The code leverages CUDA to take advantage of GPU computation, aiming to efficiently process large datasets in astrophysical studies. It is designed to facilitate the calculation of 2-point correlation functions between galaxy pairs, measuring the clustering of galaxies at different angular separations, which can provide critical insights into the underlying structures of dark matter in the universe.

### The core functionalities of the project include:
- Efficient calculation of cross-correlation histograms (DD, DR, RR)
- Leveraging parallelization for high performance using CUDA
- Running the application on GPU nodes for increased computation speed

### Technologies Used:
- CUDA for GPU acceleration
- GCC for compiling CUDA code
- SCP for secure file transfer between the local system and remote clusters

This README provides instructions to set up, compile, and run the code on **Dione**'s GPU environment.

### Prerequisites
- Access to the **Dione** cluster
- The necessary code file (`galaxy_assignment.cu`)
- Basic knowledge of **SSH** and **SCP** for file transfer

---

### Commands

```bash
# Login to Dione
ssh username@dione

# Transfer galaxy_assignment.cu from local machine to Dione
scp /path/to/galaxy_assignment.cu username@dione:/home/username/your-directory/

# Load necessary modules on Dione
module load cuda
module load GCC

# Compile the CUDA code on Dione
nvcc -arch=sm_70 galaxy_assignment.cu -o output -lm

# Run the program on the GPU
srun -p gpu -n 1 -t 10:00 --mem=1G -e err.txt -o out.txt time ./output data_100k_arcmin.txt flat_100k_arcmin.txt result.txt
