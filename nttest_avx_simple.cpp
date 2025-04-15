/*++

Module Name:

	nthash_avx.cpp

Abstract:

	Test program for the AVX implementation of ntHash.

Author:

	Roman Snytsar, October, 2018
	Microsoft AI&R

--*/
#include <string>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstring>

#include <getopt.h>
#include "nthash_avx_simple.hpp"

#define PROGRAM "nttest_avx"

static const char VERSION_MESSAGE[] =
PROGRAM " Version 1.0.0 \n"
"Written by Roman Snytsar.\n"
"Copyright 2018 Microsoft Corp\n";

static const char USAGE_MESSAGE[] =
"Usage: " PROGRAM " [OPTION]... QUERY\n"
"Report bugs to https://github.com/bcgsc/ntHash/issues\n";

namespace opt {
	unsigned kmerLen = 50;
	uint64_t nz;
	bool fastq = false;
	int window_len = 100;
	int smer_len = 31;
}

using namespace std;

static const char shortopts[] = "k:w:s:";

enum { OPT_HELP = 1, OPT_VERSION };

static const struct option longopts[] = {
	{ "kmer",	required_argument, NULL, 'k' },
	{ "windowlen",	required_argument, NULL, 'w' },
	{ "smerlen",	required_argument, NULL, 's' },
	{ "help",	no_argument, NULL, OPT_HELP },
	{ "version",	no_argument, NULL, OPT_VERSION },
	{ NULL, 0, NULL, 0 }
};

static bool debug = true;

//static const string itm[] = { "nthash", "nthash32", "ntavx2", "ntavx232", "ntavx512", "ntavx532" };
//static const string itm[] = { "nthash32", "ntavx232", "syncmer32", "syncmer32avx" };
//static const string itm[] = { "nthash", "ntavx2", "syncmer64", "syncmer64avx" };
//unsigned int nb_itm = 6; // skips ntbase 
unsigned int nb_itm = 4; // skips ntbase 
static const string itm[] = { "nthash32", "ntavx232", "syncmer32", "syncmer32avx"};

void getFtype(const char *fName) {
	std::ifstream in(fName);
	std::string hLine;
	bool good = static_cast<bool>(getline(in, hLine));
	in.close();
	if (!good) {
		std::cerr << "Error in reading file: " << fName << "\n";
		exit(EXIT_FAILURE);
	}
	if (hLine[0] == '>')
		opt::fastq = false;
	else if (hLine[0] == '@')
		opt::fastq = true;
	else {
		std::cerr << "Error in file format: " << fName << "\n";
		exit(EXIT_FAILURE);
	}
}

bool getSeq(std::ifstream &uFile, std::string &line, unsigned int &length) {
	bool good = false;
	std::string hline;
	line.clear();
	if (opt::fastq) {
		good = static_cast<bool>(getline(uFile, hline));
		good = static_cast<bool>(getline(uFile, line));
		good = static_cast<bool>(getline(uFile, hline));
		good = static_cast<bool>(getline(uFile, hline));
	}
	else {
		do {
			good = static_cast<bool>(getline(uFile, hline));
			if (hline[0] == '>' && !line.empty()) break;// !line.empty() for the first rec
			if (hline[0] != '>')line += hline;
		} while (good);
		if (!good && !line.empty())
			good = true;
	}
    length = line.length();
    if (line.length() % 16 != 0) line.insert(line.end(), 16-(line.length() % 16), 'N'); // 16-pad the line with N's because AVX512 32bits version reads nucleotides 16 by 16, don't want to read beyond bounds
	return good;
}

void hashSeqr32(const string & seq, unsigned int length) {
	uint32_t fhVal, rhVal, hVal;
	hVal = NTC31(seq.c_str(), opt::kmerLen, fhVal, rhVal);
    std::cout << std::hex << "first nthash32 " << hVal << std::endl;
	if (hVal)opt::nz++;
	for (size_t i = 1; i < length - opt::kmerLen + 1; i++) {
		hVal = NTC31(seq[i - 1], seq[i - 1 + opt::kmerLen], opt::kmerLen, fhVal, rhVal);
		if (hVal)opt::nz++;
	}
    if (debug) std::cout << std::hex << "final nthash32 " << hVal << std::endl;
}

void hashSeqr32buf(const string & seq, unsigned int length, uint32_t *buf) {
	uint32_t fhVal, rhVal, hVal;
	hVal = NTC31(seq.c_str(), opt::kmerLen, fhVal, rhVal);
    //std::cout << std::hex << "first nthash32 " << hVal << std::endl;
	buf[0] = hVal;
    //std::cout << std::hex << "first nthash32 fh " << fhVal << " rh " << rhVal << std::endl;
	for (size_t i = 1; i < length - opt::kmerLen + 1; i++) {
		hVal = NTC31(seq[i - 1], seq[i - 1 + opt::kmerLen], opt::kmerLen, fhVal, rhVal);
		buf[i] = hVal;
	}
    //if (debug) std::cout << std::hex << "final nthash32 " << hVal << std::endl;
}

void hashSeqAvx2x32buf(const string & seq, unsigned int length, uint32_t *buf) {
	const char* kmerSeq = seq.data();

	__m256i _k = _mm256_kmod31_epu32(opt::kmerLen);

	__m256i _fhVal, _rhVal, _hVal;

	_hVal = _mm256_NTC_epu32(kmerSeq, opt::kmerLen, _k, _fhVal, _rhVal);
    uint32_t hval0 = _mm256_extract_epi32(_hVal, 0);
    if (debug) std::cout << std::hex << "first hash AVX2x32 " <<  hval0 << std::endl;
        
	kmerSeq += 7;
	std::memcpy(buf, &_hVal, sizeof _hVal);

	size_t sentinel = length - opt::kmerLen + 1;

	for (size_t i = 8; i < sentinel; i += 8, kmerSeq += 8) {
		_hVal = _mm256_NTC_epu32(kmerSeq, kmerSeq + opt::kmerLen, _k, _fhVal, _rhVal);
		std::memcpy(buf+i, &_hVal, sizeof _hVal);
	}

}

void syncmer32(const string & seq, int length, int avx) {
#define BUFW 26
	opt::kmerLen = opt::smer_len;
	int window_len = opt::window_len;
	int smer_len = opt::smer_len;

	uint32_t *buf;
	int buf_len = (1 << BUFW);
	int ws = window_len-smer_len+1;
	buf = (uint32_t *)malloc((buf_len + window_len*2)*sizeof(uint32_t));
	for (int i=0; i<window_len*2; i++) buf[(1<<BUFW)+i] = 0;

	uint32_t *left_hval = (uint32_t *)malloc((window_len-smer_len+1+1)*sizeof(uint32_t));
	uint32_t *right_hval = (uint32_t *)malloc((window_len-smer_len+1+1)*sizeof(uint32_t));

	int num_syncmers = 0;

	int start = 0;
	int pos = 0;
	while (length > 0) {
		int len = (length < buf_len) ? length : buf_len;
		if (avx) {
			hashSeqAvx2x32buf(&seq[start], len+window_len*2, buf);
		} else {
			hashSeqr32buf(&seq[start], len+window_len*2, buf);
		}
		uint32_t hval;

		while (pos < len) {
#if 0
			printf("left hval ");
			for (int i=0; i<ws; i++) {
				printf("%x ", buf[pos+i]);
			}
			printf("\n");
#endif
			hval = buf[pos+ws-1];
			left_hval[ws-1] = hval;
			for (int i=ws-2; i>=0; i--) {
				if (buf[pos+i] < hval) hval = buf[pos+i];
				left_hval[i] = hval;
			}
#if 0
			printf("left min ");
			for (int i=0; i<ws; i++) {
				printf("%x ", left_hval[i]);
			}
			printf("\n");

			printf("right hval ");
			for (int i=0; i<ws; i++) {
				printf("%x ", buf[pos+ws+i]);
			}
			printf("\n");
#endif
			hval = buf[pos+ws];
			right_hval[0] = hval;
			for (int i=1; i<=ws; i++) {
				if (buf[pos+ws+i] < hval) hval = buf[pos+ws+i];
				right_hval[i] = hval;
			}
#if 0
			printf("right min ");
			for (int i=0; i<ws; i++) {
				printf("%x ", right_hval[i]);
			}
			printf("\n");
#endif
			// check syncmer for the first k-mer
			hval = left_hval[0];
			if (buf[pos] == hval || buf[pos+ws-1] == hval) {
				//printf("i=%d syncmer (%d) ", start + pos, smer_len);
				//for (int k=0; k<window_len; k++) putchar(seq[start + pos + k]);
				//printf("\n");
				num_syncmers++;
			}
			// check syncmer for the other k-mers
			for (int j=1; j<ws; j++) {
				hval = (left_hval[j] < right_hval[j-1]) ? left_hval[j] : right_hval[j-1];
				if (buf[pos+j] == hval || buf[pos+ws-1+j] == hval) {
					//printf("i=%d syncmer (%d) ", start + pos + j, smer_len);
					//for (int k=0; k<window_len; k++) putchar(seq[start + pos + j + k]);
					//printf("\n");
					num_syncmers++;
				}
	
			}
			pos += ws;
		}
		start += len;
		length -= len;
		pos -= len;
	}

	free(buf);  free(left_hval);  free(right_hval);

	printf("w=%d s=%d #syncmers %d\n", window_len, smer_len, num_syncmers);
}
#undef BUFW


void hashSeqAvx2x32(const string & seq, unsigned int length) {
	const char* kmerSeq = seq.data();

	__m256i _nz = _mm256_setzero_si256();
	__m256i _zero = _mm256_setzero_si256();

	__m256i _k = _mm256_kmod31_epu32(opt::kmerLen);

	__m256i _fhVal, _rhVal, _hVal;

	_hVal = _mm256_NTC_epu32(kmerSeq, opt::kmerLen, _k, _fhVal, _rhVal);
        
    uint32_t hval0;
    hval0 = _mm256_extract_epi32(_hVal, 0);
    if (debug) std::cout << std::hex << "first hash AVX2x32 " <<  hval0 << std::endl;

	__m256i _isZero = _mm256_cmpeq_epi32(
		_hVal,
		_zero);

	_nz = _mm256_sub_epi32(
		_nz,
		_mm256_xor_si256(
			_isZero,
			_isZero));

	kmerSeq += 7;

	size_t sentinel = length - opt::kmerLen + 1;

	for (size_t i = 8; i < sentinel; i += 8, kmerSeq += 8) {
		_hVal = _mm256_NTC_epu32(kmerSeq, kmerSeq + opt::kmerLen, _k, _fhVal, _rhVal);
	}
  
    if ((length - opt::kmerLen) % 8 == 0)
        hval0 = _mm256_extract_epi32(_hVal, 0);
    else if ((length - opt::kmerLen) % 8 == 1)
        hval0 = _mm256_extract_epi32(_hVal, 1);
    else if ((length - opt::kmerLen) % 8 == 2)
        hval0 = _mm256_extract_epi32(_hVal, 2);
    else if ((length - opt::kmerLen) % 8 == 3)
        hval0 = _mm256_extract_epi32(_hVal, 3);
    else if ((length - opt::kmerLen) % 8 == 4)
        hval0 = _mm256_extract_epi32(_hVal, 4);
    else if ((length - opt::kmerLen) % 8 == 5)
        hval0 = _mm256_extract_epi32(_hVal, 5);
    else if ((length - opt::kmerLen) % 8 == 6)
        hval0 = _mm256_extract_epi32(_hVal, 6);
    else if ((length - opt::kmerLen) % 8 == 7)
        hval0 = _mm256_extract_epi32(_hVal, 7);
    if (debug) std::cout << std::hex << "final hash AVX2x32 " <<  hval0 << std::endl;

	//printf("hVal "); print_m256i(_hVal);
}

void nthashRT(const char *readName) {
	getFtype(readName);
	cerr << "CPU time (sec) for hash algorithms for ";
	cerr << "kmer=" << opt::kmerLen << "\n";

    double times[10];
	for (unsigned method = 0; method < nb_itm; method++) {
		opt::nz = 0;
		ifstream uFile(readName);
		string line;
		clock_t sTime = 0;
        unsigned int length;
		while (getSeq(uFile, line, length)) {
            sTime = clock();
            if (itm[method] == "nthash32")
				hashSeqr32(line,length);
			else if (itm[method] == "ntavx232")
				hashSeqAvx2x32(line,length);
			else if (itm[method] == "syncmer32")
				syncmer32(line,length, 0);
			else if (itm[method] == "syncmer32avx")
				syncmer32(line,length, 1);
            times[method] += (double)(clock() - sTime) / CLOCKS_PER_SEC;
		}
		uFile.close();
    }
	for (unsigned method = 0; method < nb_itm; method++)
		cerr << itm[method] << "\t";
	cerr << "\n";
	for (unsigned method = 0; method < nb_itm; method++)
		cerr << times[method] << "\t";
	cerr << "\n";
}

int main(int argc, char** argv) {

	bool die = false;
	for (int c; (c = getopt_long(argc, argv, shortopts, longopts, NULL)) != -1;) {
		std::istringstream arg(optarg != NULL ? optarg : "");
		switch (c) {
		case '?':
			die = true;
			break;
		case 'k':
			arg >> opt::kmerLen;
			//init_kmod(opt::kmerLen);
			break;
		case 'w':
			arg >> opt::window_len;
			break;
		case 's':
			arg >> opt::smer_len;
			break;
		case OPT_HELP:
			std::cerr << USAGE_MESSAGE;
			exit(EXIT_SUCCESS);
		case OPT_VERSION:
			std::cerr << VERSION_MESSAGE;
			exit(EXIT_SUCCESS);
		}
		if (optarg != NULL && !arg.eof()) {
			std::cerr << PROGRAM ": invalid option: `-"
				<< (char)c << optarg << "'\n";
			exit(EXIT_FAILURE);
		}
	}
	if (argc - optind != 1 && argc - optind != 2) {
		std::cerr << PROGRAM ": missing arguments\n";
		die = true;
	}

	if (die) {
		std::cerr << "Try `" << PROGRAM
			<< " --help' for more information.\n";
		exit(EXIT_FAILURE);
	}

	const char *readName(argv[argc - 1]);
	opt::kmerLen = opt::smer_len;
	nthashRT(readName);

	return 0;
}
