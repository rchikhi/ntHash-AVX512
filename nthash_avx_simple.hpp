/*++

Module Name:

	nthash_avx.hpp

Abstract:

	AVX implementation of ntHash.

Author:

	Roman Snytsar, October, 2018
	Microsoft AI&R

--*/

#ifndef NT_HASH_AVX_H
#define NT_HASH_AVX_H

#include <cassert>
#include "immintrin.h"
#include "nthash_simple.hpp"

void print_m256i(__m256i vx)
{
  int x[8];
  memcpy (x, &vx, sizeof vx);
  for (int i=0; i<8; i++) {
    printf("%x ", x[i]);
  }  
}

void print_m256d(__m256i vx)
{
  long x[4];
  memcpy (x, &vx, sizeof vx);
  for (int i=0; i<4; i++) {
    printf("%lx ", x[i]);
  }  
}


// Shift vector imm bytes left across the lanes while shifting in zeroes
template <int imm>
__m256i _mm256_shift_left_si256(__m256i a) {
	__m256i c = _mm256_permute2x128_si256(a, _mm256_setzero_si256(), 0x03);
	return _mm256_alignr_epi8(a, c, 16 - imm);
}

// convert kmers 8 -> 3 bit representation, N character is mapped to 4
// 
inline __m128i _mm_CKX_epu8(const __m128i _kmerSeq) {
    // _mm_set1_epi8(a: i8) -> __m128i: Broadcasts 8-bit integer a to all elements.
	const __m128i _mask = _mm_set1_epi8(0x0f);

    // _mm_set_epi8([16 args]) -> __m128i: Sets packed 8-bit integers with the supplied values.
	const __m128i _table = _mm_set_epi8(
		4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 3, 1, 4, 0, 4);

    // _mm_shuffle_epi8(a: __m128i, b: __m128i) -> __m128i: Shuffles bytes from a according to the content of b.
	__m128i _kmer = _mm_shuffle_epi8(
		_table,
        // _mm_and_si128(a: __m128i, b: __m128i) -> __m128i: Computes the bitwise AND of 128 bits (representing integer data) in a and b.
		_mm_and_si128(
			_kmerSeq,
			_mask));

	return _kmer;
}

// encode complement of "k" modulo 31 
inline __m256i _mm256_kmod31_epu32(const uint32_t k) {
	return  _mm256_set1_epi32(31 - (k % 31));
}

// rotate 31-right bits of "_v" to the right by _s position
// elements of _s must be less than 31
inline __m256i _mm256_rorv31_epu32(const __m256i _v, const __m256i _s) {
	const __m256i _32 = _mm256_set1_epi32(32);

	return _mm256_or_si256(
            //  _mm256_srlv_epi32(a: __m256i, count: __m256i) -> __m256i: 
            //  Shifts packed 32-bit integers in a right by the amount specified by the corresponding element in count while shifting in zeros,
		_mm256_srlv_epi32(
			_v,
			_s),
        //  _mm256_srli_epi32(a: __m256i, const IMM8: i32) -> __m256i: Shifts packed 32-bit integers in a right by IMM8 while shifting in zeros
		_mm256_srli_epi32(
			_mm256_sllv_epi32(_v,
				_mm256_sub_epi32(
					_32,
					_s)),
			1));
}

// rotate 31-right bits of "_v" to the right by imm positions
template <int imm>
__m256i _mm256_rori31_epu32(const __m256i _v) {
	return _mm256_or_si256(
        // _mm256_srli_epi32(a: __m256i, const IMM8: i32) -> __m256i: Shifts packed 32-bit integers in a right by IMM8 while shifting in zeros
		_mm256_srli_epi32(
			_v,
			imm),
		_mm256_srli_epi32(
			_mm256_slli_epi32(
				_v,
				32 - imm),
			1));
}

// load kmers in 3 bit format
inline __m256i _mm256_LKX_epu32(const char * kmerSeq) {
	__m256i _kmer = _mm256_cvtepu8_epi32(
		_mm_CKX_epu8(
            //_mm_loadl_epi64(mem_addr: *const __m128i) -> __m128i: Loads 64-bit integer from memory into first element of returned vector.
            _mm_loadl_epi64(
				(__m128i const*)kmerSeq)));

	return _kmer;
}

// load forward-strand kmers
inline __m256i _mm256_LKF_epu32(const char * kmerSeq) {
	const __m256i _seed = _mm256_set_epi32(
		0, 0, 0, 0,
		(int)(seedT >> 33),
		(int)(seedG >> 33),
		(int)(seedC >> 33),
		(int)(seedA >> 33));

	__m256i _kmer = _mm256_permutevar8x32_epi32(
		_seed,
		_mm256_LKX_epu32(
			kmerSeq));

	return _kmer;
}

// load reverse-strand kmers
inline __m256i _mm256_LKR_epu32(const char * kmerSeq) {
	const __m256i _seed = _mm256_set_epi32(
		0, 0, 0, 0,
		(int)(seedA >> 33),
		(int)(seedC >> 33),
		(int)(seedG >> 33),
		(int)(seedT >> 33));

	__m256i _kmer = _mm256_permutevar8x32_epi32(
		_seed,
		_mm256_LKX_epu32(
			kmerSeq));

	return _kmer;
}

// forward-strand hash value of the base kmer, i.e. fhval(kmer_0)
inline __m256i _mm256_NTF_epu32(const char * kmerSeq, const unsigned k) {
	__m256i _hVal31 = _mm256_setzero_si256();
	//printf("i=%d _hVal31 ", 0);	print_m256i(_hVal31);

	for (unsigned i = 0; i < k; i++)
	{
		//_hVal31 = _mm256_rori31_epu32<30>(_hVal31);
		_hVal31 = _mm256_rori31_epu32<30>(_hVal31);
		//printf("i=%d _hVal31 ", i);	print_m256i(_hVal31);

		__m256i _kmer31 = _mm256_LKF_epu32(kmerSeq + i);
		//printf(" _kmer31 ");	print_m256i(_kmer31);

		_hVal31 = _mm256_xor_si256(
			_hVal31,
			_kmer31);
	}
	//printf(" _hVal31 ");	print_m256i(_hVal31);

	return _hVal31;
}

// reverse-strand hash value of the base kmer, i.e. rhval(kmer_0)
inline __m256i _mm256_NTR_epu32(const char * kmerSeq, const unsigned k, const __m256i _k) {
	const __m256i _zero = _mm256_setzero_si256();

	__m256i _hVal31 = _zero;

	for (unsigned i = 0; i < k; i++)
	{
		__m256i _kmer31 = _mm256_LKR_epu32(kmerSeq + i);

		_kmer31 = _mm256_rorv31_epu32(
			_kmer31,
			_k);

		_hVal31 = _mm256_xor_si256(
			_hVal31,
			_kmer31);

		_hVal31 = _mm256_rori31_epu32<1>(_hVal31);
	}

	return _hVal31;
}

// canonical ntHash
inline __m256i _mm256_NTC_epu32(const char * kmerSeq, const unsigned k, const __m256i _k, __m256i& _fhVal, __m256i& _rhVal) {
	_fhVal = _mm256_NTF_epu32(kmerSeq, k);
	_rhVal = _mm256_NTR_epu32(kmerSeq, k, _k);

	// _mm256_blendv_epi8(a: __m256i, b: __m256i, mask: __m256i) -> __m256i: Blends packed 8-bit integers from a and b using mask.
	__m256i _hVal = _mm256_blendv_epi8(
		_fhVal,
		_rhVal,
        // _mm256_cmpgt_epi32(a: __m256i, b: __m256i) -> __m256i: Compares packed 32-bit integers in a and b for greater-than.
		_mm256_cmpgt_epi32(
			_fhVal,
			_rhVal));

	return _hVal;
}

// forward-strand ntHash for sliding k-mers
inline __m256i _mm256_NTF_epu32(const __m256i _fhVal, const __m256i _k, const char * kmerOut, const char * kmerIn) {
	const __m256i _zero = _mm256_setzero_si256();

	// construct input kmers
	__m256i _in31 = _mm256_LKF_epu32(kmerIn);
	__m256i _out31 = _mm256_LKF_epu32(kmerOut);

	_out31 = _mm256_rorv31_epu32(
		_out31,
		_k);

	__m256i _kmer31 = _mm256_xor_si256(
		_in31,
		_out31);

	// scan-shift kmers	
	_kmer31 = _mm256_xor_si256(
		_kmer31,
		_mm256_shift_left_si256<4>(
			_mm256_rori31_epu32<30>(
				_kmer31)));

	_kmer31 = _mm256_xor_si256(
		_kmer31,
		_mm256_shift_left_si256<8>(
			_mm256_rori31_epu32<29>(
				_kmer31)));

	_kmer31 = _mm256_xor_si256(
		_kmer31,
		_mm256_permute2x128_si256(
			_mm256_rori31_epu32<27>(
				_kmer31),
			_zero,
			0x08));

	// var-shift the hash
	__m256i _hVal31 = _mm256_permutevar8x32_epi32(
		_fhVal,
		_mm256_set1_epi32(7));

	const __m256i _shift31 = _mm256_set_epi32(
		23, 24, 25, 26, 27, 28, 29, 30);

	_hVal31 = _mm256_rorv31_epu32(
		_hVal31,
		_shift31);

	// merge everything together
	_hVal31 = _mm256_xor_si256(
		_hVal31,
		_kmer31);

	return _hVal31;
}

// reverse-complement ntHash for sliding k-mers
inline __m256i _mm256_NTR_epu32(const __m256i _rhVal, const __m256i _k, const char * kmerOut, const char * kmerIn) {
	const __m256i _zero = _mm256_setzero_si256();

	// construct input kmers
	__m256i _in31 = _mm256_LKR_epu32(kmerIn);

	_in31 = _mm256_rorv31_epu32(
		_in31,
		_k);

	__m256i _out31 = _mm256_LKR_epu32(kmerOut);

	__m256i _kmer31 = _mm256_xor_si256(
		_in31,
		_out31);

	// scan-shift kmers	
	_kmer31 = _mm256_xor_si256(
		_kmer31,
		_mm256_shift_left_si256<4>(
			_mm256_rori31_epu32<1>(
				_kmer31)));

	_kmer31 = _mm256_xor_si256(
		_kmer31,
		_mm256_shift_left_si256<8>(
			_mm256_rori31_epu32<2>(
				_kmer31)));

	_kmer31 = _mm256_xor_si256(
		_kmer31,
		_mm256_permute2x128_si256(
			_mm256_rori31_epu32<4>(
				_kmer31),
			_zero,
			0x08));

	// var-shift the hash
	__m256i _hVal31 = _mm256_permutevar8x32_epi32(
		_rhVal,
		_mm256_set1_epi32(7));

	const __m256i _shift31 = _mm256_set_epi32(
		7, 6, 5, 4, 3, 2, 1, 0);

	_hVal31 = _mm256_rorv31_epu32(
		_hVal31,
		_shift31);

	// merge everything together
	_hVal31 = _mm256_xor_si256(
		_hVal31,
		_kmer31);

	_hVal31 = _mm256_rori31_epu32<1>(_hVal31);

	return _hVal31;
}

// canonical ntHash for sliding k-mers
inline __m256i _mm256_NTC_epu32(const char * kmerOut, const char * kmerIn, const __m256i _k, __m256i& _fhVal, __m256i& _rhVal) {
	_fhVal = _mm256_NTF_epu32(_fhVal, _k, kmerOut, kmerIn);
	_rhVal = _mm256_NTR_epu32(_rhVal, _k, kmerOut, kmerIn);

	__m256i _hVal = _mm256_blendv_epi8(
		_fhVal,
		_rhVal,
		_mm256_cmpgt_epi32(
			_fhVal,
			_rhVal));

	return _hVal;
}


#endif
