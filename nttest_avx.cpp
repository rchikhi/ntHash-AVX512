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

#include <getopt.h>
#include "nthash_avx.hpp"

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
}

using namespace std;

static const char shortopts[] = "k:";

enum { OPT_HELP = 1, OPT_VERSION };

static const struct option longopts[] = {
	{ "kmer",	required_argument, NULL, 'k' },
	{ "help",	no_argument, NULL, OPT_HELP },
	{ "version",	no_argument, NULL, OPT_VERSION },
	{ NULL, 0, NULL, 0 }
};

static bool debug = true;

static const string itm[] = { "nthash", "nthash32", "ntavx2", "ntavx232", "ntavx512", "ntavx532" };

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

void hashSeqb(const string & seq, unsigned int length) {
	for (size_t i = 0; i < length - opt::kmerLen + 1; i++) {
		if (NTC64(seq.c_str() + i, opt::kmerLen)) opt::nz++;
	}
}

void hashSeqr(const string & seq, unsigned int length) {
	uint64_t fhVal, rhVal, hVal;
	hVal = NTC64(seq.c_str(), opt::kmerLen, fhVal, rhVal);
    if (debug) std::cout << std::hex << "first nthash " << hVal << std::endl;
	if (hVal)opt::nz++;
	for (size_t i = 1; i < length - opt::kmerLen + 1; i++) {
		hVal = NTC64(seq[i - 1], seq[i - 1 + opt::kmerLen], opt::kmerLen, fhVal, rhVal);
		if (hVal)opt::nz++;
        if (i > length - opt::kmerLen - 15) // some debug
        {
            //std::cout << "next char " << seq[i - 1 + opt::kmerLen] << " interm hash " << hVal << std::endl;
        }
	}
    if (debug) std::cout << std::hex << "final nthash " << hVal << std::endl;
}

void hashSeqr32(const string & seq, unsigned int length) {
	uint32_t fhVal, rhVal, hVal;
	hVal = NTC32(seq.c_str(), opt::kmerLen, fhVal, rhVal);
    std::cout << std::hex << "first nthash32 " << hVal << std::endl;
    std::cout << std::hex << "first nthash32 fh " << fhVal << " rh " << rhVal << std::endl;
	if (hVal)opt::nz++;
	for (size_t i = 1; i < length - opt::kmerLen + 1; i++) {
		hVal = NTC32(seq[i - 1], seq[i - 1 + opt::kmerLen], opt::kmerLen, fhVal, rhVal);
        std::cout << std::hex << "next " << i  << " nthash32 fh " << fhVal << " rh " << rhVal << std::endl;
		if (hVal)opt::nz++;
        if (i > length - opt::kmerLen - 15) // some debug
        {
           // std::cout << "next char " << seq[i - 1 + opt::kmerLen] << " interm hash " << hVal << std::endl;
        }
	}
    if (debug) std::cout << std::hex << "final nthash32 " << hVal << std::endl;
}

void hashSeqAvx2(const string & seq, unsigned int length) {
	const char* kmerSeq = seq.data();

	__m256i _nz = _mm256_setzero_si256();
	__m256i _zero = _mm256_setzero_si256();

	__m256i _k = _mm256_kmod3133_epu64(opt::kmerLen);

	__m256i _fhVal, _rhVal, _hVal;

	_hVal = _mm256_NTC_epu64(kmerSeq, opt::kmerLen, _k, _fhVal, _rhVal);

   uint64_t hval0 = _mm256_extract_epi64(_hVal, 0);
   if (debug) std::cout << std::hex<< "first hash AVX2 " <<  hval0 << std::endl;

	__m256i _isZero = _mm256_cmpeq_epi64(
		_hVal,
		_zero);

	_nz = _mm256_sub_epi64(
		_nz,
		_mm256_xor_si256(
			_isZero,
			_isZero));

	kmerSeq += 3;

	size_t sentinel = length - opt::kmerLen + 1;

	for (size_t i = 4; i < sentinel; i += 4, kmerSeq += 4) {
		_hVal = _mm256_NTC_epu64(kmerSeq, kmerSeq + opt::kmerLen, _k, _fhVal, _rhVal);

		__m256i _isZero = _mm256_cmpeq_epi64(
			_hVal,
			_zero);

		_nz = _mm256_sub_epi64(
			_nz,
			_mm256_xor_si256(
				_isZero,
				_isZero));
            
        //hval0 = _mm256_extract_epi64(_hVal, 0);
        //if (i > sentinel-10)
        //std::cout << std::hex << "interm hash AVX2 " <<  hval0 << std::endl;
    }

	opt::nz =
		_mm256_extract_epi64(_nz, 0) +
		_mm256_extract_epi64(_nz, 1) +
		_mm256_extract_epi64(_nz, 2) +
		_mm256_extract_epi64(_nz, 3);

   if ((length - opt::kmerLen) % 4 == 0)
       hval0 = _mm256_extract_epi64(_hVal, 0);
   else if ((length - opt::kmerLen) % 4 == 1)
       hval0 = _mm256_extract_epi64(_hVal, 1);
   else if ((length - opt::kmerLen) % 4 == 2)
       hval0 = _mm256_extract_epi64(_hVal, 2);
   else if ((length - opt::kmerLen) % 4 == 3)
       hval0 = _mm256_extract_epi64(_hVal, 3);
   if (debug) std::cout << std::hex << "final hash AVX2 " <<  hval0 << std::endl;
}

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

		__m256i _isZero = _mm256_cmpeq_epi32(
			_hVal,
			_zero);

		_nz = _mm256_sub_epi32(
			_nz,
			_mm256_xor_si256(
				_isZero,
				_isZero));
	}

	opt::nz =
		_mm256_extract_epi32(_nz, 0) +
		_mm256_extract_epi32(_nz, 1) +
		_mm256_extract_epi32(_nz, 2) +
		_mm256_extract_epi32(_nz, 3) +
		_mm256_extract_epi32(_nz, 4) +
		_mm256_extract_epi32(_nz, 5) +
		_mm256_extract_epi32(_nz, 6) +
		_mm256_extract_epi32(_nz, 7);
   
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
}

void hashSeqAvx512(const string & seq, unsigned int length) {
	const char* kmerSeq = seq.data();

	__m512i _nz = _mm512_setzero_si512();
	__m512i _zero = _mm512_setzero_si512();

	__m512i _k = _mm512_kmod3133_epu64(opt::kmerLen);

	__m512i _fhVal, _rhVal, _hVal;

	_hVal = _mm512_NTC_epu64(kmerSeq, opt::kmerLen, _k, _fhVal, _rhVal);

	__m256i lo = _mm512_extracti64x4_epi64(_hVal, 0);
    uint64_t hval0 = _mm256_extract_epi64(lo, 0);
    if (debug) std::cout << std::hex << "first hash AVX512 " <<  hval0  << std::endl;

    
    __mmask8 _isZero = _mm512_cmpeq_epi64_mask(
		_hVal,
		_zero);

	_nz = _mm512_mask_sub_epi64(
		_nz,
		_isZero,
		_nz,
		_mm512_xor_epi64(
			_zero,
			_zero));

	kmerSeq += 7;

	size_t sentinel = length - opt::kmerLen + 1;

	for (size_t i = 8; i < sentinel; i += 8, kmerSeq += 8) {
		_hVal = _mm512_NTC_epu64(kmerSeq, kmerSeq + opt::kmerLen, _k, _fhVal, _rhVal);

		__mmask8 _isZero = _mm512_cmpeq_epi64_mask(
			_hVal,
			_zero);

		_nz = _mm512_mask_sub_epi64(
			_nz,
			_isZero,
			_nz,
			_mm512_xor_epi64(
				_zero,
				_zero));
	}

   opt::nz = _mm512_reduce_add_epi64(_nz);

   if ((length - opt::kmerLen) % 8 < 4)
    lo = _mm512_extracti64x4_epi64(_hVal, 0);
   else
    lo = _mm512_extracti64x4_epi64(_hVal, 1);
   if ((length - opt::kmerLen) % 4 == 0)
       hval0 = _mm256_extract_epi64(lo, 0);
   else if ((length - opt::kmerLen) % 4 == 1)
       hval0 = _mm256_extract_epi64(lo, 1);
   else if ((length - opt::kmerLen) % 4 == 2)
       hval0 = _mm256_extract_epi64(lo, 2);
   else if ((length - opt::kmerLen) % 4 == 3)
       hval0 = _mm256_extract_epi64(lo, 3);
   if (debug) std::cout << std::hex << "final hash AVX512 " <<  hval0  << std::endl;
}

void hashSeqAvx512x32(const string & seq, unsigned int length) {
	const char* kmerSeq = seq.data();

	__m512i _nz = _mm512_setzero_si512();
	__m512i _zero = _mm512_setzero_si512();

	__m512i _k = _mm512_kmod31_epu32(opt::kmerLen);

	__m512i _fhVal, _rhVal, _hVal;

	_hVal = _mm512_NTC_epu32(kmerSeq, opt::kmerLen, _k, _fhVal, _rhVal);
    
    uint32_t hval0;
    __m256i lo;
    lo = _mm512_extracti64x4_epi64(_hVal, 0);
    hval0 = _mm256_extract_epi32(lo, 0);
    if (debug) std::cout << std::hex << "first hash AVX512x32 " <<  hval0 << std::endl;

	__mmask16 _isZero = _mm512_cmpeq_epi32_mask(
		_hVal,
		_zero);

	_nz = _mm512_mask_sub_epi32(
		_nz,
		_isZero,
		_nz,
		_mm512_xor_epi32(
			_zero,
			_zero));

	kmerSeq += 15;

	size_t sentinel = length - opt::kmerLen + 1;

	for (size_t i = 16; i < sentinel; i += 16, kmerSeq += 16) {
		_hVal = _mm512_NTC_epu32(kmerSeq, kmerSeq + opt::kmerLen, _k, _fhVal, _rhVal);

		__mmask16 _isZero = _mm512_cmpeq_epi32_mask(
			_hVal,
			_zero);

		_nz = _mm512_mask_sub_epi32(
			_nz,
			_isZero,
			_nz,
			_mm512_xor_epi32(
				_zero,
				_zero));
	}

	opt::nz = _mm512_reduce_add_epi32(_nz);

    if ((length - opt::kmerLen) % 16 < 8)
        lo = _mm512_extracti64x4_epi64(_hVal, 0);
    else
        lo = _mm512_extracti64x4_epi64(_hVal, 1);
    if ((length - opt::kmerLen) % 8 == 0)
        hval0 = _mm256_extract_epi32(lo, 0);
    else if ((length - opt::kmerLen) % 8 == 1)
        hval0 = _mm256_extract_epi32(lo, 1);
    else if ((length - opt::kmerLen) % 8 == 2)
        hval0 = _mm256_extract_epi32(lo, 2);
    else if ((length - opt::kmerLen) % 8 == 3)
        hval0 = _mm256_extract_epi32(lo, 3);
    else if ((length - opt::kmerLen) % 8 == 4)
        hval0 = _mm256_extract_epi32(lo, 4);
    else if ((length - opt::kmerLen) % 8 == 5)
        hval0 = _mm256_extract_epi32(lo, 5);
    else if ((length - opt::kmerLen) % 8 == 6)
        hval0 = _mm256_extract_epi32(lo, 6);
    else if ((length - opt::kmerLen) % 8 == 7)
        hval0 = _mm256_extract_epi32(lo, 7);
   if (debug) std::cout << std::hex << "final hash AVX512x32 " <<  hval0 << std::endl;

}

void nthashRT(const char *readName) {
	getFtype(readName);
	cerr << "CPU time (sec) for hash algorithms for ";
	cerr << "kmer=" << opt::kmerLen << "\n";

    unsigned int nb_itm = 6; // skips ntbase 
    double times[10];
	for (unsigned method = 0; method < nb_itm; method++) {
		opt::nz = 0;
		ifstream uFile(readName);
		string line;
		clock_t sTime = clock();
        unsigned int length;
		while (getSeq(uFile, line, length)) {
			if (itm[method] == "nthash")
				hashSeqr(line,length);
            else if (itm[method] == "nthash32")
				hashSeqr32(line,length);
			else if (itm[method] == "ntavx2")
				hashSeqAvx2(line,length);
			else if (itm[method] == "ntavx512")
				hashSeqAvx512(line,length);
			else if (itm[method] == "ntavx232")
				hashSeqAvx2x32(line,length);
			else if (itm[method] == "ntavx532")
				hashSeqAvx512x32(line,length);
			else if (itm[method] == "ntbase")
				hashSeqb(line,length);
		}
        times[method] = (double)(clock() - sTime) / CLOCKS_PER_SEC;
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

	nthashRT(readName);

	return 0;
}
