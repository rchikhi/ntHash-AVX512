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
#include "nthash.hpp"

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

// encode complement of "k" modulo 31 and 33 into lo/hi parts of the 64 bits
inline __m256i _mm256_kmod3133_epu64(const uint64_t k) {
	return _mm256_packus_epi32(
		_mm256_set1_epi64x(31 - (k % 31)),
		_mm256_set1_epi64x(33 - (k % 33)));
}

// split "_v" into 31-bit and 33-bit parts and store parts in the rightmost bits
inline void _mm256_split3133_epu64(const __m256i _v, __m256i& _part31, __m256i& _part33) {
	_part31 = _mm256_srli_epi64(
		_v,
		33);

	_part33 = _mm256_srli_epi64(
		_mm256_slli_epi64(
			_v,
			31),
		31);
}

// merge 31-bit and 33-bit parts stored in the rightmost bits
inline __m256i _mm256_merge3133_epu64(const __m256i _part31, const __m256i _part33) {
	return _mm256_or_si256(
		_part33,
		_mm256_slli_epi64(
			_part31,
			33));
}

// rotate 31-right bits of "_v" to the right by _s position
// elements of _s must be less than 31
inline __m256i _mm256_rorv31_epu64(const __m256i _v, const __m256i _s) {
	const __m256i _64 = _mm256_set1_epi64x(64ll);

	return _mm256_or_si256(
		_mm256_srlv_epi64(
			_v,
			_s),
		_mm256_srli_epi64(
			_mm256_sllv_epi64(_v,
				_mm256_sub_epi64(
					_64,
					_s)),
			33));
}

// rotate 31-right bits of "_v" to the right by imm positions
template <int imm>
__m256i _mm256_rori31_epu64(const __m256i _v) {
	return _mm256_or_si256(
		_mm256_srli_epi64(
			_v,
			imm),
		_mm256_srli_epi64(
			_mm256_slli_epi64(
				_v,
				64 - imm),
			33));
}

// rotate 33-right bits of "_v" to the right by _s position
// elements of _s must be less than 31
inline __m256i _mm256_rorv33_epu64(const __m256i _v, const __m256i _s) {
	const __m256i _64 = _mm256_set1_epi64x(64ll);

	return _mm256_or_si256(
		_mm256_srlv_epi64(
			_v,
			_s),
		_mm256_srli_epi64(
			_mm256_sllv_epi64(_v,
				_mm256_sub_epi64(
					_64,
					_s)),
			31));
}

// rotate 33-right bits of "_v" to the right by imm positions
template <int imm>
__m256i _mm256_rori33_epu64(const __m256i _v) {
	return _mm256_or_si256(
		_mm256_srli_epi64(
			_v,
			imm),
		_mm256_srli_epi64(
			_mm256_slli_epi64(
				_v,
				64 - imm),
			31));
}

// load forward-strand kmers
inline __m256i _mm256_LKF_epu64(const char * kmerSeq) {
    // _mm256_i32gather_epi64(slice: *const i64, offsets: __m128i, const SCALE: i32) -> __m256i
    // Returns values from slice at offsets determined by offsets * scale, where scale should be 1, 2, 4 or 8.
	__m256i _kmer = _mm256_i32gather_epi64(
		(const long long*)seedTab,
        // _mm_cvtepu8_epi32(a: __m128i) -> __m128i: Zeroes extend packed unsigned 8-bit integers in a to packed 32-bit integers
		_mm_cvtepu8_epi32(
            //_mm_cvtsi32_si128(a: i32) -> __m128i: Returns a vector whose lowest element is a and all higher elements are 0.
            _mm_cvtsi32_si128(
				*(uint32_t const*) kmerSeq)),
		8);

	return _kmer;
}

// load reverse-strand kmers
inline __m256i _mm256_LKR_epu64(const char * kmerSeq) {
	__m256i _kmer = _mm256_i32gather_epi64(
		(const long long*)seedTab,
		_mm_cvtepu8_epi32(
			_mm_and_si128(
                _mm_cvtsi32_si128(
                    *(uint32_t const*) kmerSeq),
				_mm_set1_epi8(cpOff))),
		8);

	return _kmer;
}

// forward-strand hash value of the base kmer, i.e. fhval(kmer_0)
inline __m256i _mm256_NTF_epu64(const char * kmerSeq, const unsigned k) {
	__m256i _hVal31 = _mm256_setzero_si256();
	__m256i _hVal33 = _mm256_setzero_si256();

	for (unsigned i = 0; i < k; i++)
	{
		_hVal31 = _mm256_rori31_epu64<30>(_hVal31);
		_hVal33 = _mm256_rori33_epu64<32>(_hVal33);

		__m256i _kmer31, _kmer33;
		_mm256_split3133_epu64(
			_mm256_LKF_epu64(kmerSeq + i),
			_kmer31,
			_kmer33);

		_hVal31 = _mm256_xor_si256(
			_hVal31,
			_kmer31);

		_hVal33 = _mm256_xor_si256(
			_hVal33,
			_kmer33);
	}

	__m256i _hVal = _mm256_merge3133_epu64(
		_hVal31,
		_hVal33);

	return _hVal;
}

// reverse-strand hash value of the base kmer, i.e. rhval(kmer_0)
inline __m256i _mm256_NTR_epu64(const char * kmerSeq, const unsigned k, const __m256i _k) {
	const __m256i _zero = _mm256_setzero_si256();

	const __m256i _k31 = _mm256_unpacklo_epi32(
		_k,
		_zero);

	const __m256i _k33 = _mm256_unpackhi_epi32(
		_k,
		_zero);

	__m256i _hVal31 = _zero;
	__m256i _hVal33 = _zero;

	for (unsigned i = 0; i < k; i++)
	{
		__m256i _kmer31, _kmer33;
		_mm256_split3133_epu64(
			_mm256_LKR_epu64(kmerSeq + i),
			_kmer31,
			_kmer33);

		_kmer31 = _mm256_rorv31_epu64(
			_kmer31,
			_k31);

		_kmer33 = _mm256_rorv33_epu64(
			_kmer33,
			_k33);

		_hVal31 = _mm256_xor_si256(
			_hVal31,
			_kmer31);

		_hVal33 = _mm256_xor_si256(
			_hVal33,
			_kmer33);

		_hVal31 = _mm256_rori31_epu64<1>(_hVal31);
		_hVal33 = _mm256_rori33_epu64<1>(_hVal33);
	}

	__m256i _hVal = _mm256_merge3133_epu64(
		_hVal31,
		_hVal33);

	return _hVal;
}

// canonical ntHash
inline __m256i _mm256_NTC_epu64(const char * kmerSeq, const unsigned k, const __m256i _k, __m256i& _fhVal, __m256i& _rhVal) {
	_fhVal = _mm256_NTF_epu64(kmerSeq, k);
	_rhVal = _mm256_NTR_epu64(kmerSeq, k, _k);

	const __m256i _mask = _mm256_set1_epi64x(0x8000000000000000ll);

	__m256i _hVal = _mm256_blendv_epi8(
		_fhVal,
		_rhVal,
		_mm256_cmpgt_epi64(
			_mm256_xor_si256(
				_mask,
				_fhVal),
			_mm256_xor_si256(
				_mask,
				_rhVal)));

	return _hVal;
}

// canonical ntBase
inline __m256i _mm256_NTC_epu64(const char * kmerSeq, const unsigned k, const __m256i _k) {
	__m256i _fhVal, _rhVal;

	return _mm256_NTC_epu64(kmerSeq, k, _k, _fhVal, _rhVal);
}

// forward-strand ntHash for sliding k-mers
inline __m256i _mm256_NTF_epu64(const __m256i _fhVal, const __m256i _k, const char * kmerOut, const char * kmerIn) {
	const __m256i _zero = _mm256_setzero_si256();

	// construct input kmers
	__m256i _in31, _in33;
	_mm256_split3133_epu64(
		_mm256_LKF_epu64(kmerIn),
		_in31,
		_in33);

	__m256i _out31, _out33;
	_mm256_split3133_epu64(
		_mm256_LKF_epu64(kmerOut),
		_out31,
		_out33);

	_out31 = _mm256_rorv31_epu64(
		_out31,
		_mm256_unpacklo_epi32(
			_k,
			_zero));

	_out33 = _mm256_rorv33_epu64(
		_out33,
		_mm256_unpackhi_epi32(
			_k,
			_zero));

	__m256i _kmer31 = _mm256_xor_si256(
		_in31,
		_out31);

	__m256i _kmer33 = _mm256_xor_si256(
		_in33,
		_out33);

	// scan-shift kmers	
	_kmer31 = _mm256_xor_si256(
		_kmer31,
		_mm256_shift_left_si256<8>(
			_mm256_rori31_epu64<30>(
				_kmer31)));

	_kmer33 = _mm256_xor_si256(
		_kmer33,
		_mm256_shift_left_si256<8>(
			_mm256_rori33_epu64<32>(
				_kmer33)));

	_kmer31 = _mm256_xor_si256(
		_kmer31,
		_mm256_permute2x128_si256(
			_mm256_rori31_epu64<29>(
				_kmer31),
			_zero,
			0x08));

	_kmer33 = _mm256_xor_si256(
		_kmer33,
		_mm256_permute2x128_si256(
			_mm256_rori33_epu64<31>(
				_kmer33),
			_zero,
			0x08));

	// var-shift the hash
    // _mm256_permute4x64_epi64(a: __m256i, const IMM8: i32) -> __m256i:
    // Permutes 64-bit integers from a using control mask imm8.
	__m256i _hVal = _mm256_permute4x64_epi64(
		_fhVal,
		0xff);

	__m256i _hVal31, _hVal33;
	_mm256_split3133_epu64(
		_hVal,
		_hVal31,
		_hVal33);

	const __m256i _shift31 = _mm256_set_epi64x(
		27ll, 28ll, 29ll, 30ll);

	_hVal31 = _mm256_rorv31_epu64(
		_hVal31,
		_shift31);

	const __m256i _shift33 = _mm256_set_epi64x(
		29ll, 30ll, 31ll, 32ll);

	_hVal33 = _mm256_rorv33_epu64(
		_hVal33,
		_shift33);

	// merge everything together
	_hVal31 = _mm256_xor_si256(
		_hVal31,
		_kmer31);

	_hVal33 = _mm256_xor_si256(
		_hVal33,
		_kmer33);

	_hVal = _mm256_merge3133_epu64(
		_hVal31,
		_hVal33);

	return _hVal;
}

// reverse-complement ntHash for sliding k-mers
inline __m256i _mm256_NTR_epu64(const __m256i _rhVal, const __m256i _k, const char * kmerOut, const char * kmerIn) {
	const __m256i _zero = _mm256_setzero_si256();

	// construct input kmers
	__m256i _in31, _in33;
	_mm256_split3133_epu64(
		_mm256_LKR_epu64(kmerIn),
		_in31,
		_in33);

	_in31 = _mm256_rorv31_epu64(
		_in31,
		_mm256_unpacklo_epi32(
			_k,
			_zero));

	_in33 = _mm256_rorv33_epu64(
		_in33,
		_mm256_unpackhi_epi32(
			_k,
			_zero));

	__m256i _out31, _out33;
	_mm256_split3133_epu64(
		_mm256_LKR_epu64(kmerOut),
		_out31,
		_out33);

	__m256i _kmer31 = _mm256_xor_si256(
		_in31,
		_out31);

	__m256i _kmer33 = _mm256_xor_si256(
		_in33,
		_out33);

	// scan-shift kmers	
	_kmer31 = _mm256_xor_si256(
		_kmer31,
		_mm256_shift_left_si256<8>(
			_mm256_rori31_epu64<1>(
				_kmer31)));

	_kmer33 = _mm256_xor_si256(
		_kmer33,
		_mm256_shift_left_si256<8>(
			_mm256_rori33_epu64<1>(
				_kmer33)));

	_kmer31 = _mm256_xor_si256(
		_kmer31,
		_mm256_permute2x128_si256(
			_mm256_rori31_epu64<2>(
				_kmer31),
			_zero,
			0x08));

	_kmer33 = _mm256_xor_si256(
		_kmer33,
		_mm256_permute2x128_si256(
			_mm256_rori33_epu64<2>(
				_kmer33),
			_zero,
			0x08));

	// var-shift the hash
	__m256i _hVal = _mm256_permute4x64_epi64(
		_rhVal,
		0xff);

	__m256i _hVal31, _hVal33;
	_mm256_split3133_epu64(
		_hVal,
		_hVal31,
		_hVal33);

	const __m256i _shift = _mm256_set_epi64x(
		3ll, 2ll, 1ll, 0ll);

	_hVal31 = _mm256_rorv31_epu64(
		_hVal31,
		_shift);

	_hVal33 = _mm256_rorv33_epu64(
		_hVal33,
		_shift);

	// merge everything together
	_hVal31 = _mm256_xor_si256(
		_hVal31,
		_kmer31);

	_hVal33 = _mm256_xor_si256(
		_hVal33,
		_kmer33);

	_hVal31 = _mm256_rori31_epu64<1>(_hVal31);
	_hVal33 = _mm256_rori33_epu64<1>(_hVal33);

	_hVal = _mm256_merge3133_epu64(
		_hVal31,
		_hVal33);

	return _hVal;
}

// canonical ntHash for sliding k-mers
inline __m256i _mm256_NTC_epu64(const char * kmerOut, const char * kmerIn, const __m256i _k, __m256i& _fhVal, __m256i& _rhVal) {
	_fhVal = _mm256_NTF_epu64(_fhVal, _k, kmerOut, kmerIn);
	_rhVal = _mm256_NTR_epu64(_rhVal, _k, kmerOut, kmerIn);

	__m256i _mask = _mm256_set1_epi64x(0x8000000000000000ll);

	__m256i _hVal = _mm256_blendv_epi8(
		_fhVal,
		_rhVal,
		_mm256_cmpgt_epi64(
			_mm256_xor_si256(
				_mask,
				_fhVal),
			_mm256_xor_si256(
				_mask,
				_rhVal)));

	return _hVal;
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

	for (unsigned i = 0; i < k; i++)
	{
		_hVal31 = _mm256_rori31_epu32<30>(_hVal31);

		__m256i _kmer31 = _mm256_LKF_epu32(kmerSeq + i);

		_hVal31 = _mm256_xor_si256(
			_hVal31,
			_kmer31);
	}

	return _hVal31;
}

// forward-strand hash value of the base kmer, i.e. fhval(kmer_0)
// ntHash1 version
inline __m256i _mm256_NTF_epu32_ntHash1(const char * kmerSeq, const unsigned k) {
	__m256i _hVal32 = _mm256_setzero_si256();
	const __m256i _mask32 = _mm256_set1_epi64x(0xFFFFFFFFl);
    //uint64_t hval0;

	for (unsigned i = 0; i < k; i++)
	{
        //_hVal31 = _mm256_rori31_epu32<30>(_hVal31);
        // for that 32 bits version, I do like the 32 bits scalar implementation in nthash.hpp: 
        // just a roll 1 left (it's ntHash1 with a k>32 periodicity, even worse than k>64)
        _hVal32 = _mm256_or_si256(
                _mm256_slli_epi64(
                    _hVal32,
                    1),
                _mm256_srli_epi64(
                    _hVal32,
                    31));

		__m256i _kmer32 = _mm256_LKF_epu32(kmerSeq + i);
        
		_hVal32 = _mm256_xor_si256(
			_hVal32,
			_kmer32);

		_hVal32 = _mm256_and_si256(
			_hVal32,
			_mask32);

        //hval0 = _mm256_extract_epi64(_hVal32, 0);
        //std::cout << std::hex << i << " first hash AVX2x32 " <<  hval0 << std::endl;
	}

	return _hVal32;
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

// reverse-strand hash value of the base kmer, i.e. rhval(kmer_0)
// ntHash1 version
inline __m256i _mm256_NTR_epu32_nthash1(const char * kmerSeq, const unsigned k, const __m256i _k) {
    (void)_k;
	const __m256i _zero = _mm256_setzero_si256();
	const __m256i _mask32 = _mm256_set1_epi64x(0xFFFFFFFFl);

	__m256i _hVal32 = _zero;

	for (unsigned i = 0; i < k; i++)
	{
        // same: do ntHash1 like nthash.hpp for this 32bits version
        _hVal32 = _mm256_or_si256(
                _mm256_slli_epi64(
                    _hVal32,
                    1),
                _mm256_srli_epi64(
                    _hVal32,
                    31));

		__m256i _kmer32 = _mm256_LKR_epu32(kmerSeq + k - 1 - i);

		//_kmer31 = _mm256_rorv31_epu32(
		//	_kmer31,
		//	_k);

		_hVal32 = _mm256_xor_si256(
			_hVal32,
			_kmer32);

		//_hVal31 = _mm256_rori31_epu32<1>(_hVal31);

		_hVal32 = _mm256_and_si256(
			_hVal32,
			_mask32);

        //uint64_t hval0 = _mm256_extract_epi64(_hVal32, 0);
        //std::cout << std::hex << "first hash rev AVX2x32 " <<  hval0 << std::endl;
	}

	return _hVal32;
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

// canonical ntBase
inline __m256i _mm256_NTC_epu32(const char * kmerSeq, const unsigned k, const __m256i _k) {
	__m256i _fhVal, _rhVal;

	return _mm256_NTC_epu32(kmerSeq, k, _k, _fhVal, _rhVal);
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

// forward-strand ntHash for sliding k-mers (32 bits, ntHash1)
// this is a work in progress to transform this function to the nthash1 flavor. I didn't finish it. So this function is incorrect for now.
inline __m256i _mm256_NTF_epu32_nthash1_wip(const __m256i _fhVal, const __m256i _k, const char * kmerOut, const char * kmerIn) {
	const __m256i _zero = _mm256_setzero_si256();
	const __m256i _32 = _mm256_set1_epi32(32);

	// construct input kmers
	__m256i _in32 = _mm256_LKF_epu32(kmerIn);

	__m256i _out32 = _mm256_LKF_epu32(kmerOut);

    _out32 = _mm256_or_si256(
            //  _mm256_srlv_epi32(a: __m256i, count: __m256i) -> __m256i: 
            //  Shifts packed 32-bit integers in a right by the amount specified by the corresponding element in count while shifting in zeros,
            _mm256_sllv_epi32(
                _out32,
                _k),
            //  _mm256_srli_epi32(a: __m256i, const IMM8: i32) -> __m256i: Shifts packed 32-bit integers in a right by IMM8 while shifting in zeros
            _mm256_sllv_epi32(_out32,
                _mm256_sub_epi32(
                    _32,
                    _k)));
    
	__m256i _kmer31 = _mm256_xor_si256(
		_in32,
		_out32);

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

// encode complement of "k" modulo 31 and 33 into lo/hi parts of the 64 bits
inline __m512i _mm512_kmod3133_epu64(const uint64_t k) {
	return _mm512_packus_epi32(
		_mm512_set1_epi64(31 - (k % 31)),
		_mm512_set1_epi64(33 - (k % 33)));
}

// split "_v" into 31-bit and 33-bit parts and store parts in the rightmost bits
inline void _mm512_split3133_epu64(const __m512i _v, __m512i& _part31, __m512i& _part33) {
	_part31 = _mm512_srli_epi64(
		_v,
		33);

	_part33 = _mm512_srli_epi64(
		_mm512_slli_epi64(
			_v,
			31),
		31);
}

// merge 31-bit and 33-bit parts stored in the rightmost bits
inline __m512i _mm512_merge3133_epu64(const __m512i _part31, const __m512i _part33) {
	return _mm512_or_si512(
		_part33,
		_mm512_slli_epi64(
			_part31,
			33));
}

// rotate 31-right bits of "_v" to the right by _s position
// elements of _s must be less than 31
inline __m512i _mm512_rorv31_epu64(const __m512i _v, const __m512i _s) {
	const __m512i _64 = _mm512_set1_epi64(64ll);

	return _mm512_or_si512(
		_mm512_srlv_epi64(
			_v,
			_s),
		_mm512_srli_epi64(
			_mm512_sllv_epi64(_v,
				_mm512_sub_epi64(
					_64,
					_s)),
			33));
}

// rotate 31-right bits of "_v" to the right by imm positions
template <int imm>
__m512i _mm512_rori31_epu64(const __m512i _v) {
	return _mm512_or_si512(
		_mm512_srli_epi64(
			_v,
			imm),
		_mm512_srli_epi64(
			_mm512_slli_epi64(
				_v,
				64 - imm),
			33));
}

// rotate 33-right bits of "_v" to the right by _s position
// elements of _s must be less than 31
inline __m512i _mm512_rorv33_epu64(const __m512i _v, const __m512i _s) {
	const __m512i _64 = _mm512_set1_epi64(64ll);

	return _mm512_or_si512(
		_mm512_srlv_epi64(
			_v,
			_s),
		_mm512_srli_epi64(
			_mm512_sllv_epi64(_v,
				_mm512_sub_epi64(
					_64,
					_s)),
			31));
}

// rotate 33-right bits of "_v" to the right by imm positions
template <int imm>
__m512i _mm512_rori33_epu64(const __m512i _v) {
	return _mm512_or_si512(
		_mm512_srli_epi64(
			_v,
			imm),
		_mm512_srli_epi64(
			_mm512_slli_epi64(
				_v,
				64 - imm),
			31));
}

/*
// load kmers in 3 bit format
inline __m512i _mm512_LKX_epu64(const char * kmerSeq) {
	__m512i _kmer = _mm512_cvtepu8_epi64(
		_mm_CKX_epu8(
            _mm_maskz_load_epi64(
                0x01,
                kmerSeq)));

	return _kmer;
}
*/

// previous non-crashing version
inline __m512i _mm512_LKX_epu64(const char * kmerSeq) {
    // _mm512_cvtepu8_epi64(a: __m128i) -> __m512i: Zero extend packed unsigned 8-bit integers in the low 8 bytes of a to packed 64-bit integers, and store the results in dst.
	__m512i _kmer = _mm512_cvtepu8_epi64(
			_mm_CKX_epu8(
				_mm_loadl_epi64(
					(__m128i const*)kmerSeq)));

	return _kmer;
}

// load forward-strand kmers
inline __m512i _mm512_LKF_epu64(const char * kmerSeq) {
	const __m512i _seed = _mm512_set_epi64(
		0, 0, 0, 0,
		seedT, seedG, seedC, seedA);

    //_mm512_permutexvar_epi32(idx: __m512i, a: __m512i) -> __m512i: Shuffle 32-bit integers in a across lanes using the corresponding index in idx, and store the results in dst.
	__m512i _kmer = _mm512_permutexvar_epi64(
		_mm512_LKX_epu64(
			kmerSeq),
		_seed);

	return _kmer;
}

// load reverse-strand kmers
inline __m512i _mm512_LKR_epu64(const char * kmerSeq) {
	const __m512i _seed = _mm512_set_epi64(
		0, 0, 0, 0,
		seedA, seedC, seedG, seedT);

	__m512i _kmer = _mm512_permutexvar_epi64(
		_mm512_LKX_epu64(
			kmerSeq),
		_seed);

	return _kmer;
}

// forward-strand hash value of the base kmer, i.e. fhval(kmer_0)
inline __m512i _mm512_NTF_epu64(const char * kmerSeq, const unsigned k) {
	__m512i _hVal31 = _mm512_setzero_si512();
	__m512i _hVal33 = _mm512_setzero_si512();

	for (unsigned i = 0; i < k; i++)
	{
		_hVal31 = _mm512_rori31_epu64<30>(_hVal31);
		_hVal33 = _mm512_rori33_epu64<32>(_hVal33);

		__m512i _kmer31, _kmer33;
		_mm512_split3133_epu64(
			_mm512_LKF_epu64(kmerSeq + i),
			_kmer31,
			_kmer33);

		_hVal31 = _mm512_xor_epi64(
			_hVal31,
			_kmer31);

		_hVal33 = _mm512_xor_epi64(
			_hVal33,
			_kmer33);
	}

	__m512i _hVal = _mm512_merge3133_epu64(
		_hVal31,
		_hVal33);

	return _hVal;
}

// reverse-strand hash value of the base kmer, i.e. rhval(kmer_0)
inline __m512i _mm512_NTR_epu64(const char * kmerSeq, const unsigned k, const __m512i _k) {
	const __m512i _zero = _mm512_setzero_si512();

	const __m512i _k31 = _mm512_unpacklo_epi32(
		_k,
		_zero);

	const __m512i _k33 = _mm512_unpackhi_epi32(
		_k,
		_zero);

	__m512i _hVal31 = _zero;
	__m512i _hVal33 = _zero;

	for (unsigned i = 0; i < k; i++)
	{
		__m512i _kmer31, _kmer33;
		_mm512_split3133_epu64(
			_mm512_LKR_epu64(kmerSeq + i),
			_kmer31,
			_kmer33);

		_kmer31 = _mm512_rorv31_epu64(
			_kmer31,
			_k31);

		_kmer33 = _mm512_rorv33_epu64(
			_kmer33,
			_k33);

		_hVal31 = _mm512_xor_epi64(
			_hVal31,
			_kmer31);

		_hVal33 = _mm512_xor_epi64(
			_hVal33,
			_kmer33);

		_hVal31 = _mm512_rori31_epu64<1>(_hVal31);
		_hVal33 = _mm512_rori33_epu64<1>(_hVal33);
	}

	__m512i _hVal = _mm512_merge3133_epu64(
		_hVal31,
		_hVal33);

	return _hVal;
}

// canonical ntHash
inline __m512i _mm512_NTC_epu64(const char * kmerSeq, const unsigned k, const __m512i _k, __m512i& _fhVal, __m512i& _rhVal) {
	_fhVal = _mm512_NTF_epu64(kmerSeq, k);
	_rhVal = _mm512_NTR_epu64(kmerSeq, k, _k);

	__m512i _hVal = _mm512_mask_blend_epi64(
		_mm512_cmpgt_epu64_mask(
			_fhVal,
			_rhVal),
		_fhVal,
		_rhVal);

	return _hVal;
}

// canonical ntBase
inline __m512i _mm512_NTC_epu64(const char * kmerSeq, const unsigned k, const __m512i _k) {
	__m512i _fhVal, _rhVal;

	return _mm512_NTC_epu64(kmerSeq, k, _k, _fhVal, _rhVal);
}

// forward-strand ntHash for sliding k-mers
inline __m512i _mm512_NTF_epu64(const __m512i _fhVal, const __m512i _k, const char * kmerOut, const char * kmerIn) {
	const __m512i _zero = _mm512_setzero_si512();

	// construct input kmers
	__m512i _in31, _in33;
	_mm512_split3133_epu64(
		_mm512_LKF_epu64(kmerIn),
		_in31,
		_in33);

	__m512i _out31, _out33;
	_mm512_split3133_epu64(
		_mm512_LKF_epu64(kmerOut),
		_out31,
		_out33);

	_out31 = _mm512_rorv31_epu64(
		_out31,
		_mm512_unpacklo_epi32(
			_k,
			_zero));

	_out33 = _mm512_rorv33_epu64(
		_out33,
		_mm512_unpackhi_epi32(
			_k,
			_zero));

	__m512i _kmer31 = _mm512_xor_epi64(
		_in31,
		_out31);

	__m512i _kmer33 = _mm512_xor_epi64(
		_in33,
		_out33);

	// scan-shift kmers	
	_kmer31 = _mm512_xor_epi64(
		_kmer31,
		_mm512_maskz_expand_epi64(
			0xfe,
			_mm512_rori31_epu64<30>(
				_kmer31)));

	_kmer33 = _mm512_xor_epi64(
		_kmer33,
		_mm512_maskz_expand_epi64(
			0xfe,
			_mm512_rori33_epu64<32>(
				_kmer33)));

	_kmer31 = _mm512_xor_epi64(
		_kmer31,
		_mm512_maskz_expand_epi64(
			0xfc,
			_mm512_rori31_epu64<29>(
				_kmer31)));

	_kmer33 = _mm512_xor_epi64(
		_kmer33,
		_mm512_maskz_expand_epi64(
			0xfc,
			_mm512_rori33_epu64<31>(
				_kmer33)));

	_kmer31 = _mm512_xor_epi64(
		_kmer31,
		_mm512_maskz_expand_epi64(
			0xf0,
			_mm512_rori31_epu64<27>(
				_kmer31)));

	_kmer33 = _mm512_xor_epi64(
		_kmer33,
		_mm512_maskz_expand_epi64(
			0xf0,
			_mm512_rori33_epu64<29>(
				_kmer33)));

	// var-shift the hash
	__m512i _hVal = _mm512_permutexvar_epi64(
		_mm512_set1_epi64(7),
		_fhVal);

	__m512i _hVal31, _hVal33;
	_mm512_split3133_epu64(
		_hVal,
		_hVal31,
		_hVal33);

	const __m512i _shift31 = _mm512_set_epi64(
		23ll, 24ll, 25ll, 26ll, 27ll, 28ll, 29ll, 30ll);

	_hVal31 = _mm512_rorv31_epu64(
		_hVal31,
		_shift31);

	const __m512i _shift33 = _mm512_set_epi64(
		25ll, 26ll, 27ll, 28ll, 29ll, 30ll, 31ll, 32ll);

	_hVal33 = _mm512_rorv33_epu64(
		_hVal33,
		_shift33);

	// merge everything together
	_hVal31 = _mm512_xor_epi64(
		_hVal31,
		_kmer31);

	_hVal33 = _mm512_xor_epi64(
		_hVal33,
		_kmer33);

	_hVal = _mm512_merge3133_epu64(
		_hVal31,
		_hVal33);

	return _hVal;
}

// reverse-complement ntHash for sliding k-mers
inline __m512i _mm512_NTR_epu64(const __m512i _rhVal, const __m512i _k, const char * kmerOut, const char * kmerIn) {
	const __m512i _zero = _mm512_setzero_si512();

	// construct input kmers
	__m512i _in31, _in33;
	_mm512_split3133_epu64(
		_mm512_LKR_epu64(kmerIn),
		_in31,
		_in33);

	_in31 = _mm512_rorv31_epu64(
		_in31,
		_mm512_unpacklo_epi32(
			_k,
			_zero));

	_in33 = _mm512_rorv33_epu64(
		_in33,
		_mm512_unpackhi_epi32(
			_k,
			_zero));

	__m512i _out31, _out33;
	_mm512_split3133_epu64(
		_mm512_LKR_epu64(kmerOut),
		_out31,
		_out33);

	__m512i _kmer31 = _mm512_xor_epi64(
		_in31,
		_out31);

	__m512i _kmer33 = _mm512_xor_epi64(
		_in33,
		_out33);

	// scan-shift kmers	
	_kmer31 = _mm512_xor_epi64(
		_kmer31,
		_mm512_maskz_expand_epi64(
			0xfe,
			_mm512_rori31_epu64<1>(
				_kmer31)));

	_kmer33 = _mm512_xor_epi64(
		_kmer33,
		_mm512_maskz_expand_epi64(
			0xfe,
			_mm512_rori33_epu64<1>(
				_kmer33)));

	_kmer31 = _mm512_xor_epi64(
		_kmer31,
		_mm512_maskz_expand_epi64(
			0xfc,
			_mm512_rori31_epu64<2>(
				_kmer31)));

	_kmer33 = _mm512_xor_epi64(
		_kmer33,
		_mm512_maskz_expand_epi64(
			0xfc,
			_mm512_rori33_epu64<2>(
				_kmer33)));

	_kmer31 = _mm512_xor_epi64(
		_kmer31,
		_mm512_maskz_expand_epi64(
			0xf0,
			_mm512_rori31_epu64<4>(
				_kmer31)));

	_kmer33 = _mm512_xor_epi64(
		_kmer33,
		_mm512_maskz_expand_epi64(
			0xf0,
			_mm512_rori33_epu64<4>(
				_kmer33)));

	// var-shift the hash
	__m512i _hVal = _mm512_permutexvar_epi64(
		_mm512_set1_epi64(7),
		_rhVal);

	__m512i _hVal31, _hVal33;
	_mm512_split3133_epu64(
		_hVal,
		_hVal31,
		_hVal33);

	const __m512i _shift = _mm512_set_epi64(
		7ll, 6ll, 5ll, 4ll, 3ll, 2ll, 1ll, 0ll);

	_hVal31 = _mm512_rorv31_epu64(
		_hVal31,
		_shift);

	_hVal33 = _mm512_rorv33_epu64(
		_hVal33,
		_shift);

	// merge everything together
	_hVal31 = _mm512_xor_epi64(
		_hVal31,
		_kmer31);

	_hVal33 = _mm512_xor_epi64(
		_hVal33,
		_kmer33);

	_hVal31 = _mm512_rori31_epu64<1>(_hVal31);
	_hVal33 = _mm512_rori33_epu64<1>(_hVal33);

	_hVal = _mm512_merge3133_epu64(
		_hVal31,
		_hVal33);

	return _hVal;
}

// canonical ntHash for sliding k-mers
inline __m512i _mm512_NTC_epu64(const char * kmerOut, const char * kmerIn, const __m512i _k, __m512i& _fhVal, __m512i& _rhVal) {
	_fhVal = _mm512_NTF_epu64(_fhVal, _k, kmerOut, kmerIn);
	_rhVal = _mm512_NTR_epu64(_rhVal, _k, kmerOut, kmerIn);

	__m512i _hVal = _mm512_mask_blend_epi64(
		_mm512_cmpgt_epu64_mask(
			_fhVal,
			_rhVal),
		_fhVal,
		_rhVal);

	return _hVal;
}


// encode complement of "k" modulo 31 
inline __m512i _mm512_kmod31_epu32(const uint32_t k) {
	return  _mm512_set1_epi32(31 - (k % 31));
}

// rotate 31-right bits of "_v" to the right by _s position
// elements of _s must be less than 31
inline __m512i _mm512_rorv31_epu32(const __m512i _v, const __m512i _s) {
	const __m512i _32 = _mm512_set1_epi32(32);

	return _mm512_or_epi32(
		_mm512_srlv_epi32(
			_v,
			_s),
		_mm512_srli_epi32(
			_mm512_sllv_epi32(_v,
				_mm512_sub_epi32(
					_32,
					_s)),
			1));
}

// rotate 31-right bits of "_v" to the right by imm positions
template <int imm>
__m512i _mm512_rori31_epu32(const __m512i _v) {
	return _mm512_or_epi32(
		_mm512_srli_epi32(
			_v,
			imm),
		_mm512_srli_epi32(
			_mm512_slli_epi32(
				_v,
				32 - imm),
			1));
}

// load kmers in 3 bit format
inline __m512i _mm512_LKX_epu32(const char * kmerSeq) {
	__m512i _kmer = _mm512_cvtepu8_epi32(
		_mm_CKX_epu8(
			_mm_loadu_si128(
				(const __m128i *)kmerSeq)));

	return _kmer;
}

// load forward-strand kmers
inline __m512i _mm512_LKF_epu32(const char * kmerSeq) {
	const __m512i _seed = _mm512_set_epi32(
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
		(int)(seedT >> 33),
		(int)(seedG >> 33),
		(int)(seedC >> 33),
		(int)(seedA >> 33));

	__m512i _kmer = _mm512_permutexvar_epi32(
		_mm512_LKX_epu32(
			kmerSeq),
		_seed);

	return _kmer;
}

// load reverse-strand kmers
inline __m512i _mm512_LKR_epu32(const char * kmerSeq) {
	const __m512i _seed = _mm512_set_epi32(
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
		(int)(seedA >> 33),
		(int)(seedC >> 33),
		(int)(seedG >> 33),
		(int)(seedT >> 33));

	__m512i _kmer = _mm512_permutexvar_epi32(
		_mm512_LKX_epu32(
			kmerSeq),
		_seed);

	return _kmer;
}

// forward-strand hash value of the base kmer, i.e. fhval(kmer_0)
inline __m512i _mm512_NTF_epu32(const char * kmerSeq, const unsigned k) {
	__m512i _hVal31 = _mm512_setzero_si512();

	for (unsigned i = 0; i < k; i++)
	{
		_hVal31 = _mm512_rori31_epu32<30>(_hVal31);

		__m512i _kmer31 = _mm512_LKF_epu32(kmerSeq + i);

		_hVal31 = _mm512_xor_epi32(
			_hVal31,
			_kmer31);
	}

	return _hVal31;
}

// reverse-strand hash value of the base kmer, i.e. rhval(kmer_0)
inline __m512i _mm512_NTR_epu32(const char * kmerSeq, const unsigned k, const __m512i _k) {
	const __m512i _zero = _mm512_setzero_si512();

	__m512i _hVal31 = _zero;

	for (unsigned i = 0; i < k; i++)
	{
		__m512i _kmer31 = _mm512_LKR_epu32(kmerSeq + i);

		_kmer31 = _mm512_rorv31_epu32(
			_kmer31,
			_k);

		_hVal31 = _mm512_xor_epi32(
			_hVal31,
			_kmer31);

		_hVal31 = _mm512_rori31_epu32<1>(_hVal31);
	}

	return _hVal31;
}

// canonical ntHash
inline __m512i _mm512_NTC_epu32(const char * kmerSeq, const unsigned k, const __m512i _k, __m512i& _fhVal, __m512i& _rhVal) {
	_fhVal = _mm512_NTF_epu32(kmerSeq, k);
	_rhVal = _mm512_NTR_epu32(kmerSeq, k, _k);

	__m512i _hVal = _mm512_mask_blend_epi32(
		_mm512_cmpgt_epu32_mask(
			_fhVal,
			_rhVal),
		_fhVal,
		_rhVal);
	return _hVal;
}

// canonical ntBase
inline __m512i _mm512_NTC_epu32(const char * kmerSeq, const unsigned k, const __m512i _k) {
	__m512i _fhVal, _rhVal;

	return _mm512_NTC_epu32(kmerSeq, k, _k, _fhVal, _rhVal);
}

// forward-strand ntHash for sliding k-mers
inline __m512i _mm512_NTF_epu32(const __m512i _fhVal, const __m512i _k, const char * kmerOut, const char * kmerIn) {
	//const __m512i _zero = _mm512_setzero_si512();

	// construct input kmers
	__m512i _in31 = _mm512_LKF_epu32(kmerIn);

	__m512i _out31 = _mm512_LKF_epu32(kmerOut);

	_out31 = _mm512_rorv31_epu32(
		_out31,
		_k);

	__m512i _kmer31 = _mm512_xor_epi32(
		_in31,
		_out31);

	// scan-shift kmers	
	_kmer31 = _mm512_xor_epi32(
		_kmer31,
		_mm512_maskz_expand_epi32(
			0xfffe,
			_mm512_rori31_epu32<30>(
				_kmer31)));

	_kmer31 = _mm512_xor_epi32(
		_kmer31,
		_mm512_maskz_expand_epi32(
			0xfffc,
			_mm512_rori31_epu32<29>(
				_kmer31)));

	_kmer31 = _mm512_xor_epi32(
		_kmer31,
		_mm512_maskz_expand_epi32(
			0xfff0,
			_mm512_rori31_epu32<27>(
				_kmer31)));

	_kmer31 = _mm512_xor_epi32(
		_kmer31,
		_mm512_maskz_expand_epi32(
			0xff00,
			_mm512_rori31_epu32<23>(
				_kmer31)));

	// var-shift the hash
	__m512i _hVal31 = _mm512_permutexvar_epi32(
		_mm512_set1_epi32(15),
		_fhVal);

	const __m512i _shift31 = _mm512_set_epi32(
		15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30);

	_hVal31 = _mm512_rorv31_epu32(
		_hVal31,
		_shift31);

	// merge everything together
	_hVal31 = _mm512_xor_epi32(
		_hVal31,
		_kmer31);

	return _hVal31;
}

// reverse-complement ntHash for sliding k-mers
inline __m512i _mm512_NTR_epu32(const __m512i _rhVal, const __m512i _k, const char * kmerOut, const char * kmerIn) {
	//const __m512i _zero = _mm512_setzero_si512();

	// construct input kmers
	__m512i _in31 = _mm512_LKR_epu32(kmerIn);

	_in31 = _mm512_rorv31_epu32(
		_in31,
		_k);

	__m512i _out31 = _mm512_LKR_epu32(kmerOut);

	__m512i _kmer31 = _mm512_xor_epi32(
		_in31,
		_out31);

	// scan-shift kmers	
	_kmer31 = _mm512_xor_epi32(
		_kmer31,
		_mm512_maskz_expand_epi32(
			0xfffe,
			_mm512_rori31_epu32<1>(
				_kmer31)));

	_kmer31 = _mm512_xor_epi32(
		_kmer31,
		_mm512_maskz_expand_epi32(
			0xfffc,
			_mm512_rori31_epu32<2>(
				_kmer31)));

	_kmer31 = _mm512_xor_epi32(
		_kmer31,
		_mm512_maskz_expand_epi32(
			0xfff0,
			_mm512_rori31_epu32<4>(
				_kmer31)));

	_kmer31 = _mm512_xor_epi32(
		_kmer31,
		_mm512_maskz_expand_epi32(
			0xff00,
			_mm512_rori31_epu32<8>(
				_kmer31)));

	// var-shift the hash
	__m512i _hVal31 = _mm512_permutexvar_epi32(
		_mm512_set1_epi32(15),
		_rhVal);

	const __m512i _shift31 = _mm512_set_epi32(
		15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

	_hVal31 = _mm512_rorv31_epu32(
		_hVal31,
		_shift31);

	// merge everything together
	_hVal31 = _mm512_xor_epi32(
		_hVal31,
		_kmer31);

	_hVal31 = _mm512_rori31_epu32<1>(_hVal31);

	return _hVal31;
}

// canonical ntHash for sliding k-mers
inline __m512i _mm512_NTC_epu32(const char * kmerOut, const char * kmerIn, const __m512i _k, __m512i& _fhVal, __m512i& _rhVal) {
	_fhVal = _mm512_NTF_epu32(_fhVal, _k, kmerOut, kmerIn);
	_rhVal = _mm512_NTR_epu32(_rhVal, _k, kmerOut, kmerIn);

	__m512i _hVal = _mm512_mask_blend_epi32(
		_mm512_cmpgt_epu32_mask(
			_fhVal,
			_rhVal),
		_fhVal,
		_rhVal);

	return _hVal;
}

#endif
