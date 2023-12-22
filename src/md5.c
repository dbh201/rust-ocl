kernel void getmd5(global uchar *sums, global const uchar *blocks, global const size_t *sizePtr) {
	private int gid = get_global_id(0);
	const size_t size = sizePtr[gid];
	private uint16 M = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
	for(int i = 0; i < size>>2; i++) {
		//M[i] = blocks[gid*64 + i*4]<<24 | blocks[gid*64 + i*4 + 1]<<16 | blocks[gid*64 + i*4 + 2]<<8 | blocks[gid*64 + i*4 + 3]; 	//big endian
		M[i] = blocks[gid*64 + i*4] | blocks[gid*64 + i*4 + 1]<<8 | blocks[gid*64 + i*4 + 2]<<16 | blocks[gid*64 + i*4 + 3]<<24; 	//little endian
	}
	const int last = size - (size%4);
	for(int i = 0; i < size%4; i++) {
		M[size>>2] |= blocks[gid*64 + last + i]<<(i*8); //little endian
	}
	__constant unsigned int s[64] = {
	7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,
	5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,
	4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,
	6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21
	};
	__constant unsigned int K[64] = {
	0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 
	0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
	0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
	0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
	0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
	0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
	0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
	0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
	0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
	0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
	0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
	0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
	0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
	0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
	0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
	0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391
	};

	// init state
	unsigned int a0 = 0x67452301;   // A
	unsigned int b0 = 0xefcdab89;   // B
	unsigned int c0 = 0x98badcfe;   // C
	unsigned int d0 = 0x10325476;   // D

	// Pre-processing: adding a single 1 bit
	M[size>>2] |= 0x80<<((size%4)*8);

	// Notice: the input bytes are considered as bit strings,
	//  where the first bit is the most significant bit of the byte.[53]

	// Pre-processing: padding with zeros
	// PS: this implementation will never exceed a single block, so padding is already done.
	unsigned int bits = (size*8);
	M[14] = bits;
	// Initialize hash value for this chunk:
	unsigned int A = a0;
	unsigned int B = b0;
	unsigned int C = c0;
	unsigned int D = d0;

	for(int i = 0; i < 64; i++) {
		unsigned int F;
		size_t g;
		switch(i>>4) {
			case 0:
				F = (B & C) | ((~B) & D);
				g = i;
				break;
			case 1:
				F = (D & B) | ((~D) & C);
				g = ((5*i) + 1) % 16;
				break;
			case 2:
				F = (B ^ C ^ D);
				g = ((3*i) + 5) % 16;
				break;
			case 3:
				F = (C ^ (B | (~D)));
				g = (7*i) % 16;
				break;
		}
		//printf("---iter %i---",i);
		//printf("A: %08x B: %08x C: %08x D: %08x",A,B,C,D);
		//printf("F: %08x M[%02d]: %08x K[%02d]: %08x s[%02d]: %02d",F,g,M[g],i,K[i],i,s[i]);
		F = F + A + K[i] + M[g];
		A = D;
		D = C;
		C = B;
		B = B + rotate(F,s[i]);
	}

	//printf("%08x + %08x = ",a0,A);
	a0 += A;
	//printf("%08x",a0);
	//printf("%08x + %08x = ",b0,B);
	b0 += B;
	//printf("%08x",b0);
	//printf("%08x + %08x = ",c0,C);
	c0 += C;
	//printf("%08x",c0);
	//printf("%08x + %08x = ",d0,D);
	d0 += D;
	//printf("%08x",d0);

	for(int i = 3; i >= 0; i--) {
		sums[gid*16 + 3-i] = a0 & 0xFF;
		a0 >>= 8;
		sums[gid*16 + 7-i] = b0 & 0xFF;
		b0 >>= 8;
		sums[gid*16 + 11-i] = c0 & 0xFF;
		c0 >>= 8;
		sums[gid*16 + 15-i] = d0 & 0xFF;
		d0 >>= 8;
	}
	// need to fit everything into a printf so that it all prints at the same time
	printf("gid: % 2i, size: % 2i\tblock: %08v16x\n%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x\n",gid,size,M,
		sums[gid*16],
		sums[gid*16 + 1],
		sums[gid*16 + 2],
		sums[gid*16 + 3],
		sums[gid*16 + 4],
		sums[gid*16 + 5],
		sums[gid*16 + 6],
		sums[gid*16 + 7],
		sums[gid*16 + 8],
		sums[gid*16 + 9],
		sums[gid*16 + 10],
		sums[gid*16 + 11],
		sums[gid*16 + 12],
		sums[gid*16 + 13],
		sums[gid*16 + 14],
		sums[gid*16 + 15] );
}
