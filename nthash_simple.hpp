/*
 *
 * nthash.hpp
 * Author: Hamid Mohamadi
 * Genome Sciences Centre,
 * British Columbia Cancer Agency
 */

#ifndef NT_HASH_H
#define NT_HASH_H

#include <stdint.h>

// offset for the complement base in the random seeds table
const uint8_t cpOff = 0x07;

// 64-bit random seeds corresponding to bases and their complements
static const uint64_t seedA = 0x3c8bfbb395c60474;
static const uint64_t seedC = 0x3193c18562a02b4c;
static const uint64_t seedG = 0x20323ed082572324;
static const uint64_t seedT = 0x295549f54be24456;
static const uint64_t seedN = 0x0000000000000000;

// 32-bit random seeds corresponding to bases and their complements
static const uint32_t seed32A = (int)(seedA >> 33);
static const uint32_t seed32C = (int)(seedC >> 33);
static const uint32_t seed32G = (int)(seedG >> 33);
static const uint32_t seed32T = (int)(seedT >> 33);
static const uint32_t seed32N = 0x00000000;

static const uint32_t seedTab32[256] = {
    seed32N, seed32T, seed32N, seed32G, seed32A, seed32N, seed32N, seed32C, // 0..7
    seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, // 8..15
    seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, // 16..23
    seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, // 24..31
    seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, // 32..39
    seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, // 40..47
    seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, // 48..55
    seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, // 56..63
    seed32N, seed32A, seed32N, seed32C, seed32N, seed32N, seed32N, seed32G, // 64..71
    seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, // 72..79
    seed32N, seed32N, seed32N, seed32N, seed32T, seed32N, seed32N, seed32N, // 80..87
    seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, // 88..95
    seed32N, seed32A, seed32N, seed32C, seed32N, seed32N, seed32N, seed32G, // 96..103
    seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, // 104..111
    seed32N, seed32N, seed32N, seed32N, seed32T, seed32N, seed32N, seed32N, // 112..119
    seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, // 120..127
    seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, // 128..135
    seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, // 136..143
    seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, // 144..151
    seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, // 152..159
    seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, // 160..167
    seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, // 168..175
    seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, // 176..183
    seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, // 184..191
    seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, // 192..199
    seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, // 200..207
    seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, // 208..215
    seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, // 216..223
    seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, // 224..231
    seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, // 232..239
    seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, // 240..247
    seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N, seed32N  // 248..255
};


// rotate 31-left bits of "v" to the left by "s" positions
inline uint64_t rol31(const uint64_t v, unsigned s) {
    s%=31;
    return ((v << s) | (v >> (31 - s))) & 0x7FFFFFFF;
}

inline uint32_t NTF31(const char * kmerSeq, const unsigned k) {
    uint32_t hVal=0;
    for(unsigned i=0; i<k; i++) {
        hVal = rol31(hVal, 1);
        hVal ^= seedTab32[(unsigned char)kmerSeq[i]];

        //std::cout << std::hex << i << " first nthash32 " << hVal << std::endl;
    }
    return hVal & 0x7fffffff;
}

inline uint32_t NTR31(const char * kmerSeq, const unsigned k) {
    uint32_t hVal=0;
    for(unsigned i=0; i<k; i++) {
        hVal = rol31(hVal, 1);
        hVal ^= seedTab32[(unsigned char)kmerSeq[k-1-i]&cpOff];
        //std::cout << std::hex << i << " first nthash32 rev " << hVal << std::endl;
    }
    return hVal & 0x7fffffff;
}

inline uint32_t NTF31(const uint32_t fhVal, const unsigned k, const unsigned char charOut, const unsigned char charIn) {
    uint32_t hVal = rol31(fhVal, 1);
    hVal ^= seedTab32[charIn];
    uint32_t sOut = (seedTab32[charOut] << k) | (seedTab32[charOut] >> (31-k)) ;
    hVal ^= sOut;
    return hVal & 0x7fffffff;
}

inline uint32_t NTR31(const uint32_t rhVal, const unsigned k, const unsigned char charOut, const unsigned char charIn) {
    uint32_t sIn = rol31(seedTab32[charIn&cpOff],k-1);
    uint32_t hVal = rol31(rhVal, 30);
    hVal = hVal ^ sIn;
    hVal ^= rol31(seedTab32[charOut&cpOff], 30);
    return hVal & 0x7fffffff;
}

inline uint32_t NTC31(const char * kmerSeq, const unsigned k, uint32_t& fhVal, uint32_t& rhVal) {
    fhVal = NTF31(kmerSeq, k);
    rhVal = NTR31(kmerSeq, k);
    return (rhVal<fhVal)? rhVal : fhVal;
}

inline uint32_t NTC31(const unsigned char charOut, const unsigned char charIn, const unsigned k, uint32_t& fhVal, uint32_t& rhVal) {
    fhVal = NTF31(fhVal, k, charOut, charIn);
    rhVal = NTR31(rhVal, k, charOut, charIn);
    return (rhVal<fhVal)? rhVal : fhVal;
}


#endif
