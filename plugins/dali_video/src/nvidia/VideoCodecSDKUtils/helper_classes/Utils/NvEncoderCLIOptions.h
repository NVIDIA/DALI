/*
 * This copyright notice applies to this file only
 *
 * SPDX-FileCopyrightText: Copyright (c) 2010-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <iterator>
#include <cstring>
#include <functional>
#include "../Utils/Logger.h"

extern simplelogger::Logger *logger;

#ifndef _WIN32
inline bool operator==(const GUID &guid1, const GUID &guid2) {
    return !memcmp(&guid1, &guid2, sizeof(GUID));
}

inline bool operator!=(const GUID &guid1, const GUID &guid2) {
    return !(guid1 == guid2);
}
#endif

/*
 * Helper class for parsing generic encoder options and preparing encoder
 * initialization parameters. This class also provides some utility methods
 * which generate verbose descriptions of the provided set of encoder
 * initialization parameters.
 */
class NvEncoderInitParam {
public:
    NvEncoderInitParam(const char *szParam = "", 
        std::function<void(NV_ENC_INITIALIZE_PARAMS *pParams)> *pfuncInit = NULL, bool _bLowLatency = false) 
        : strParam(szParam), bLowLatency(_bLowLatency)
    {
        if (pfuncInit) {
            funcInit = *pfuncInit;
        }

        std::transform(strParam.begin(), strParam.end(), strParam.begin(), tolower);
        std::istringstream ss(strParam);
        tokens = std::vector<std::string> {
            std::istream_iterator<std::string>(ss),
            std::istream_iterator<std::string>() 
        };

        for (unsigned i = 0; i < tokens.size(); i++)
        {
            if (tokens[i] == "-codec" && ++i != tokens.size())
            {
                ParseString("-codec", tokens[i], vCodec, szCodecNames, &guidCodec);
                continue;
            }
            if (tokens[i] == "-preset" && ++i != tokens.size()) {
                ParseString("-preset", tokens[i], vPreset, szPresetNames, &guidPreset);
                continue;
            }
            if (tokens[i] == "-tuninginfo" && ++i != tokens.size())
            {
                ParseString("-tuninginfo", tokens[i], vTuningInfo, szTuningInfoNames, &m_TuningInfo);
                continue;
            }
        }
    }
    virtual ~NvEncoderInitParam() {}
    virtual bool IsCodecH264() {
        return GetEncodeGUID() == NV_ENC_CODEC_H264_GUID;
    }

    virtual bool IsCodecHEVC() {
        return GetEncodeGUID() == NV_ENC_CODEC_HEVC_GUID;
    }

    virtual bool IsCodecAV1() {
        return GetEncodeGUID() == NV_ENC_CODEC_AV1_GUID;
    }

    std::string GetHelpMessage(bool bMeOnly = false, bool bUnbuffered = false, bool bHide444 = false, bool bOutputInVidMem = false)
    {
        std::ostringstream oss;

        if (bOutputInVidMem && bMeOnly)
        {
            oss << "-codec       Codec: " << "h264" << std::endl;
        }
        else
        {
            oss << "-codec       Codec: " << szCodecNames << std::endl;
        }

        oss << "-preset      Preset: " << szPresetNames << std::endl
            << "-profile     H264: " << szH264ProfileNames;

        if (bOutputInVidMem && bMeOnly)
        {
            oss << std::endl;
        }
        else
        {
            oss << "; HEVC: " << szHevcProfileNames;
            oss << "; AV1: " << szAV1ProfileNames << std::endl;
        }

        if (!bMeOnly)
        {
            if (bLowLatency == false)
                oss << "-tuninginfo  TuningInfo: " << szTuningInfoNames << std::endl;
            else
                oss << "-tuninginfo  TuningInfo: " << szLowLatencyTuningInfoNames << std::endl;
            oss << "-multipass   Multipass: " << szMultipass << std::endl;
        }

        if (!bHide444 && !bLowLatency)
        {
            oss << "-444         (Only for RGB input) YUV444 encode. Not valid for AV1 Codec" << std::endl;
        }
        if (bMeOnly) return oss.str();
        oss << "-fps         Frame rate" << std::endl;

        if (!bUnbuffered && !bLowLatency)
        {
            oss << "-bf          Number of consecutive B-frames" << std::endl;
        }

        if (!bLowLatency)
        {
            oss << "-rc          Rate control mode: " << szRcModeNames << std::endl
                << "-gop         Length of GOP (Group of Pictures)" << std::endl
                << "-bitrate     Average bit rate, can be in unit of 1, K, M" << std::endl
                << "Note:        Fps or Average bit rate values for each session can be specified in the form of v1,v1,v3 (no space) for AppTransOneToN" << std::endl 
                << "             If the number of 'bitrate' or 'fps' values specified are less than the number of sessions, then the last specified value will be considered for the remaining sessions" << std::endl
                << "-maxbitrate  Max bit rate, can be in unit of 1, K, M" << std::endl
                << "-vbvbufsize  VBV buffer size in bits, can be in unit of 1, K, M" << std::endl
                << "-vbvinit     VBV initial delay in bits, can be in unit of 1, K, M" << std::endl
                << "-aq          Enable spatial AQ and set its stength (range 1-15, 0-auto)" << std::endl
                << "-temporalaq  (No value) Enable temporal AQ" << std::endl
                << "-cq          Target constant quality level for VBR mode (range 1-51, 0-auto)" << std::endl;
        }
        if (!bUnbuffered && !bLowLatency)
        {
            oss << "-lookahead   Maximum depth of lookahead (range 0-(31 - number of B frames))" << std::endl;
        }
        oss << "-qmin        Min QP value" << std::endl
            << "-qmax        Max QP value" << std::endl
            << "-initqp      Initial QP value" << std::endl;
        if (!bLowLatency)
        {
            oss << "-constqp     QP value for constqp rate control mode" << std::endl
                << "Note: QP value can be in the form of qp_of_P_B_I or qp_P,qp_B,qp_I (no space)" << std::endl;
        }
        if (bUnbuffered && !bLowLatency)
        {
            oss << "Note: Options -bf and -lookahead are unavailable for this app" << std::endl;
        }
        return oss.str();
    }

    /**
     * @brief Generate and return a string describing the values of the main/common
     *        encoder initialization parameters
     */
    std::string MainParamToString(const NV_ENC_INITIALIZE_PARAMS *pParams) {
        std::ostringstream os;
        os 
            << "Encoding Parameters:" 
            << std::endl << "\tcodec        : " << ConvertValueToString(vCodec, szCodecNames, pParams->encodeGUID)
            << std::endl << "\tpreset       : " << ConvertValueToString(vPreset, szPresetNames, pParams->presetGUID);
        if (pParams->tuningInfo)
        {
            os << std::endl << "\ttuningInfo   : " << ConvertValueToString(vTuningInfo, szTuningInfoNames, pParams->tuningInfo);
        }
        os
            << std::endl << "\tprofile      : " << ConvertValueToString(vProfile, szProfileNames, pParams->encodeConfig->profileGUID)
            << std::endl << "\tchroma       : " << ConvertValueToString(vChroma, szChromaNames, (pParams->encodeGUID == NV_ENC_CODEC_H264_GUID) ? pParams->encodeConfig->encodeCodecConfig.h264Config.chromaFormatIDC :
                                                   (pParams->encodeGUID == NV_ENC_CODEC_HEVC_GUID) ? pParams->encodeConfig->encodeCodecConfig.hevcConfig.chromaFormatIDC :
                                                   pParams->encodeConfig->encodeCodecConfig.av1Config.chromaFormatIDC)
            << std::endl << "\tbitdepth     : " << ((pParams->encodeGUID == NV_ENC_CODEC_H264_GUID) ? 0 : (pParams->encodeGUID == NV_ENC_CODEC_HEVC_GUID) ?
                                                     pParams->encodeConfig->encodeCodecConfig.hevcConfig.pixelBitDepthMinus8 : pParams->encodeConfig->encodeCodecConfig.av1Config.pixelBitDepthMinus8) + 8
            << std::endl << "\trc           : " << ConvertValueToString(vRcMode, szRcModeNames, pParams->encodeConfig->rcParams.rateControlMode)
            ;
            if (pParams->encodeConfig->rcParams.rateControlMode == NV_ENC_PARAMS_RC_CONSTQP) {
                os << " (P,B,I=" << pParams->encodeConfig->rcParams.constQP.qpInterP << "," << pParams->encodeConfig->rcParams.constQP.qpInterB << "," << pParams->encodeConfig->rcParams.constQP.qpIntra << ")";
            }
        os
            << std::endl << "\tfps          : " << pParams->frameRateNum << "/" << pParams->frameRateDen
            << std::endl << "\tgop          : " << (pParams->encodeConfig->gopLength == NVENC_INFINITE_GOPLENGTH ? "INF" : std::to_string(pParams->encodeConfig->gopLength))
            << std::endl << "\tbf           : " << pParams->encodeConfig->frameIntervalP - 1
            << std::endl << "\tmultipass    : " << pParams->encodeConfig->rcParams.multiPass
            << std::endl << "\tsize         : " << pParams->encodeWidth << "x" << pParams->encodeHeight
            << std::endl << "\tbitrate      : " << pParams->encodeConfig->rcParams.averageBitRate
            << std::endl << "\tmaxbitrate   : " << pParams->encodeConfig->rcParams.maxBitRate
            << std::endl << "\tvbvbufsize   : " << pParams->encodeConfig->rcParams.vbvBufferSize
            << std::endl << "\tvbvinit      : " << pParams->encodeConfig->rcParams.vbvInitialDelay
            << std::endl << "\taq           : " << (pParams->encodeConfig->rcParams.enableAQ ? (pParams->encodeConfig->rcParams.aqStrength ? std::to_string(pParams->encodeConfig->rcParams.aqStrength) : "auto") : "disabled")
            << std::endl << "\ttemporalaq   : " << (pParams->encodeConfig->rcParams.enableTemporalAQ ? "enabled" : "disabled")
            << std::endl << "\tlookahead    : " << (pParams->encodeConfig->rcParams.enableLookahead ? std::to_string(pParams->encodeConfig->rcParams.lookaheadDepth) : "disabled")
            << std::endl << "\tcq           : " << (unsigned int)pParams->encodeConfig->rcParams.targetQuality
            << std::endl << "\tqmin         : P,B,I=" << (int)pParams->encodeConfig->rcParams.minQP.qpInterP << "," << (int)pParams->encodeConfig->rcParams.minQP.qpInterB << "," << (int)pParams->encodeConfig->rcParams.minQP.qpIntra
            << std::endl << "\tqmax         : P,B,I=" << (int)pParams->encodeConfig->rcParams.maxQP.qpInterP << "," << (int)pParams->encodeConfig->rcParams.maxQP.qpInterB << "," << (int)pParams->encodeConfig->rcParams.maxQP.qpIntra
            << std::endl << "\tinitqp       : P,B,I=" << (int)pParams->encodeConfig->rcParams.initialRCQP.qpInterP << "," << (int)pParams->encodeConfig->rcParams.initialRCQP.qpInterB << "," << (int)pParams->encodeConfig->rcParams.initialRCQP.qpIntra
            ;
        return os.str();
    }

public:
    virtual GUID GetEncodeGUID() { return guidCodec; }
    virtual GUID GetPresetGUID() { return guidPreset; }
    virtual NV_ENC_TUNING_INFO GetTuningInfo() { return m_TuningInfo; }

    /*
     * @brief Set encoder initialization parameters based on input options
     * This method parses the tokens formed from the command line options
     * provided to the application and sets the fields from NV_ENC_INITIALIZE_PARAMS
     * based on the supplied values.
     */

    virtual void setTransOneToN(bool isTransOneToN)
    {
        bTransOneToN = isTransOneToN;
    }

    virtual void SetInitParams(NV_ENC_INITIALIZE_PARAMS *pParams, NV_ENC_BUFFER_FORMAT eBufferFormat)
    {
        NV_ENC_CONFIG &config = *pParams->encodeConfig;
        int nGOPOption = 0, nBFramesOption = 0;
        for (unsigned i = 0; i < tokens.size(); i++)
        {
            if (
                tokens[i] == "-codec"      && ++i ||
                tokens[i] == "-preset"     && ++i ||
                tokens[i] == "-tuninginfo" && ++i ||
                tokens[i] == "-multipass" && ++i != tokens.size() && ParseString("-multipass", tokens[i], vMultiPass, szMultipass, &config.rcParams.multiPass) ||
                tokens[i] == "-profile"    && ++i != tokens.size() && (IsCodecH264() ? 
                    ParseString("-profile", tokens[i], vH264Profile, szH264ProfileNames, &config.profileGUID) : IsCodecHEVC() ?
                    ParseString("-profile", tokens[i], vHevcProfile, szHevcProfileNames, &config.profileGUID) :
                    ParseString("-profile", tokens[i], vAV1Profile, szAV1ProfileNames, &config.profileGUID)) ||
                tokens[i] == "-rc"         && ++i != tokens.size() && ParseString("-rc",          tokens[i], vRcMode, szRcModeNames, &config.rcParams.rateControlMode)                    ||
                tokens[i] == "-fps"        && ++i != tokens.size() && ParseInt("-fps",            tokens[i], &pParams->frameRateNum)                                                      ||
                tokens[i] == "-bf"         && ++i != tokens.size() && ParseInt("-bf",             tokens[i], &config.frameIntervalP) && ++config.frameIntervalP && ++nBFramesOption       ||
                tokens[i] == "-bitrate"    && ++i != tokens.size() && ParseBitRate("-bitrate",    tokens[i], &config.rcParams.averageBitRate)                                             ||
                tokens[i] == "-maxbitrate" && ++i != tokens.size() && ParseBitRate("-maxbitrate", tokens[i], &config.rcParams.maxBitRate)                                                 ||
                tokens[i] == "-vbvbufsize" && ++i != tokens.size() && ParseBitRate("-vbvbufsize", tokens[i], &config.rcParams.vbvBufferSize)                                              ||
                tokens[i] == "-vbvinit"    && ++i != tokens.size() && ParseBitRate("-vbvinit",    tokens[i], &config.rcParams.vbvInitialDelay)                                            ||
                tokens[i] == "-cq"         && ++i != tokens.size() && ParseInt("-cq",             tokens[i], &config.rcParams.targetQuality)                                              ||
                tokens[i] == "-initqp"     && ++i != tokens.size() && ParseQp("-initqp",          tokens[i], &config.rcParams.initialRCQP) && (config.rcParams.enableInitialRCQP = true)  ||
                tokens[i] == "-qmin"       && ++i != tokens.size() && ParseQp("-qmin",            tokens[i], &config.rcParams.minQP) && (config.rcParams.enableMinQP = true)              ||
                tokens[i] == "-qmax"       && ++i != tokens.size() && ParseQp("-qmax",            tokens[i], &config.rcParams.maxQP) && (config.rcParams.enableMaxQP = true)              ||
                tokens[i] == "-constqp"    && ++i != tokens.size() && ParseQp("-constqp",         tokens[i], &config.rcParams.constQP)                                                    ||
                tokens[i] == "-temporalaq" && (config.rcParams.enableTemporalAQ = true)
            )
            {
                continue;
            }
            if (tokens[i] == "-lookahead" && ++i != tokens.size() && ParseInt("-lookahead", tokens[i], &config.rcParams.lookaheadDepth))
            {
                config.rcParams.enableLookahead = config.rcParams.lookaheadDepth > 0;
                continue;
            }
            int aqStrength;
            if (tokens[i] == "-aq" && ++i != tokens.size() && ParseInt("-aq", tokens[i], &aqStrength)) {
                config.rcParams.enableAQ = true;
                config.rcParams.aqStrength = aqStrength;
                continue;
            }

            if (tokens[i] == "-gop" && ++i != tokens.size() && ParseInt("-gop", tokens[i], &config.gopLength))
            {
                nGOPOption = 1;
                if (IsCodecH264()) 
                {
                    config.encodeCodecConfig.h264Config.idrPeriod = config.gopLength;
                }
                else if (IsCodecHEVC())
                {
                    config.encodeCodecConfig.hevcConfig.idrPeriod = config.gopLength;
                }
                else
                {
                    config.encodeCodecConfig.av1Config.idrPeriod = config.gopLength;
                }
                continue;
            }

            if (tokens[i] == "-444")
            {
                if (IsCodecH264()) 
                {
                    config.encodeCodecConfig.h264Config.chromaFormatIDC = 3;
                } 
                else if (IsCodecHEVC())
                {
                    config.encodeCodecConfig.hevcConfig.chromaFormatIDC = 3;
                }
                else
                {
                    std::ostringstream errmessage;
                    errmessage << "Incorrect Parameter: YUV444 Input not supported with AV1 Codec" << std::endl;
                    throw std::invalid_argument(errmessage.str());
                }
                continue;
            }

            std::ostringstream errmessage;
            errmessage << "Incorrect parameter: " << tokens[i] << std::endl;
            errmessage << "Re-run the application with the -h option to get a list of the supported options.";
            errmessage << std::endl;

            throw std::invalid_argument(errmessage.str());
        }

        if (IsCodecHEVC())
        {
            if (eBufferFormat == NV_ENC_BUFFER_FORMAT_YUV420_10BIT || eBufferFormat == NV_ENC_BUFFER_FORMAT_YUV444_10BIT)
            {
                config.encodeCodecConfig.hevcConfig.pixelBitDepthMinus8 = 2;
            }
        }

        if (IsCodecAV1())
        {
            if (eBufferFormat == NV_ENC_BUFFER_FORMAT_YUV420_10BIT)
            {
                config.encodeCodecConfig.av1Config.pixelBitDepthMinus8 = 2;
                config.encodeCodecConfig.av1Config.inputPixelBitDepthMinus8 = 2;
            }
        }

        if (nGOPOption && nBFramesOption && (config.gopLength < ((uint32_t)config.frameIntervalP)))
        {
            std::ostringstream errmessage;
            errmessage << "gopLength (" << config.gopLength << ") must be greater or equal to frameIntervalP (number of B frames + 1) (" << config.frameIntervalP << ")\n";
            throw std::invalid_argument(errmessage.str());
        }

        funcInit(pParams);
        LOG(INFO) << NvEncoderInitParam().MainParamToString(pParams);
        LOG(TRACE) << NvEncoderInitParam().FullParamToString(pParams);
    }

private:
    /*
     * Helper methods for parsing tokens (generated by splitting the command line)
     * and performing conversions to the appropriate target type/value.
     */
    template<typename T>
    bool ParseString(const std::string &strName, const std::string &strValue, const std::vector<T> &vValue, const std::string &strValueNames, T *pValue) {
        std::vector<std::string> vstrValueName = split(strValueNames, ' ');
        auto it = std::find(vstrValueName.begin(), vstrValueName.end(), strValue);
        if (it == vstrValueName.end()) {
            LOG(ERROR) << strName << " options: " << strValueNames;
            return false;
        }
        *pValue = vValue[it - vstrValueName.begin()];
        return true;
    }
    template<typename T>
    std::string ConvertValueToString(const std::vector<T> &vValue, const std::string &strValueNames, T value) {
        auto it = std::find(vValue.begin(), vValue.end(), value);
        if (it == vValue.end()) {
            LOG(ERROR) << "Invalid value. Can't convert to one of " << strValueNames;
            return std::string();
        }
        return split(strValueNames, ' ')[it - vValue.begin()];
    }
    bool ParseBitRate(const std::string &strName, const std::string &strValue, unsigned *pBitRate) {
        if(bTransOneToN)
        {
            std::vector<std::string> oneToNBitrate = split(strValue, ',');
            std::string currBitrate;
            if ((bitrateCnt + 1) > oneToNBitrate.size())
            {
                currBitrate = oneToNBitrate[oneToNBitrate.size() - 1];
            }
            else
            {
                currBitrate = oneToNBitrate[bitrateCnt];
                bitrateCnt++;
            }

            try {
                size_t l;
                double r = std::stod(currBitrate, &l);
                char c = currBitrate[l];
                if (c != 0 && c != 'k' && c != 'm') {
                    LOG(ERROR) << strName << " units: 1, K, M (lower case also allowed)";
                }
                *pBitRate = (unsigned)((c == 'm' ? 1000000 : (c == 'k' ? 1000 : 1)) * r);
            }
            catch (std::invalid_argument) {
                return false;
            }
            return true;
        }

        else
        {
            try {
                size_t l;
                double r = std::stod(strValue, &l);
                char c = strValue[l];
                if (c != 0 && c != 'k' && c != 'm') {
                    LOG(ERROR) << strName << " units: 1, K, M (lower case also allowed)";
                }
                *pBitRate = (unsigned)((c == 'm' ? 1000000 : (c == 'k' ? 1000 : 1)) * r);
            }
            catch (std::invalid_argument) {
                return false;
            }
            return true;
        }
    }
    template<typename T>
    bool ParseInt(const std::string &strName, const std::string &strValue, T *pInt) {
        if (bTransOneToN)
        {
            std::vector<std::string> oneToNFps = split(strValue, ',');
            std::string currFps;
            if ((fpsCnt + 1) > oneToNFps.size())
            {
                currFps = oneToNFps[oneToNFps.size() - 1];
            }
            else
            {
                currFps = oneToNFps[fpsCnt];
                fpsCnt++;
            }

            try {
                *pInt = std::stoi(currFps);
            }
            catch (std::invalid_argument) {
                LOG(ERROR) << strName << " need a value of positive number";
                return false;
            }
            return true;
        }
        else
        {
            try {
                *pInt = std::stoi(strValue);
            }
            catch (std::invalid_argument) {
                LOG(ERROR) << strName << " need a value of positive number";
                return false;
            }
            return true;
        }
    }
    bool ParseQp(const std::string &strName, const std::string &strValue, NV_ENC_QP *pQp) {
        std::vector<std::string> vQp = split(strValue, ',');
        try {
            if (vQp.size() == 1) {
                unsigned qp = (unsigned)std::stoi(vQp[0]);
                *pQp = {qp, qp, qp};
            } else if (vQp.size() == 3) {
                *pQp = {(unsigned)std::stoi(vQp[0]), (unsigned)std::stoi(vQp[1]), (unsigned)std::stoi(vQp[2])};
            } else {
                LOG(ERROR) << strName << " qp_for_P_B_I or qp_P,qp_B,qp_I (no space is allowed)";
                return false;
            }
        } catch (std::invalid_argument) {
            return false;
        }
        return true;
    }
    std::vector<std::string> split(const std::string &s, char delim) {
        std::stringstream ss(s);
        std::string token;
        std::vector<std::string> tokens;
        while (getline(ss, token, delim)) {
            tokens.push_back(token);
        }
        return tokens;
    }

private:
    std::string strParam;
    std::function<void(NV_ENC_INITIALIZE_PARAMS *pParams)> funcInit = [](NV_ENC_INITIALIZE_PARAMS *pParams){};
    std::vector<std::string> tokens;
    GUID guidCodec = NV_ENC_CODEC_H264_GUID;
    GUID guidPreset = NV_ENC_PRESET_P3_GUID;
    NV_ENC_TUNING_INFO m_TuningInfo = NV_ENC_TUNING_INFO_HIGH_QUALITY;
    bool bLowLatency = false;
    uint32_t bitrateCnt = 0;
    uint32_t fpsCnt = 0;
    bool bTransOneToN = 0;
    
    const char *szCodecNames = "h264 hevc av1";
    std::vector<GUID> vCodec = std::vector<GUID> {
        NV_ENC_CODEC_H264_GUID,
        NV_ENC_CODEC_HEVC_GUID,
        NV_ENC_CODEC_AV1_GUID
    };
    
    const char *szChromaNames = "yuv420 yuv444";
    std::vector<uint32_t> vChroma = std::vector<uint32_t>
    {
        1, 3
    };
    
    const char *szPresetNames = "p1 p2 p3 p4 p5 p6 p7";
    std::vector<GUID> vPreset = std::vector<GUID> {
        NV_ENC_PRESET_P1_GUID,
        NV_ENC_PRESET_P2_GUID,
        NV_ENC_PRESET_P3_GUID,
        NV_ENC_PRESET_P4_GUID,
        NV_ENC_PRESET_P5_GUID,
        NV_ENC_PRESET_P6_GUID,
        NV_ENC_PRESET_P7_GUID,
    };

    const char *szH264ProfileNames = "baseline main high high444";
    std::vector<GUID> vH264Profile = std::vector<GUID> {
        NV_ENC_H264_PROFILE_BASELINE_GUID,
        NV_ENC_H264_PROFILE_MAIN_GUID,
        NV_ENC_H264_PROFILE_HIGH_GUID,
        NV_ENC_H264_PROFILE_HIGH_444_GUID,
    };
    const char *szHevcProfileNames = "main main10 frext";
    std::vector<GUID> vHevcProfile = std::vector<GUID> {
        NV_ENC_HEVC_PROFILE_MAIN_GUID,
        NV_ENC_HEVC_PROFILE_MAIN10_GUID,
        NV_ENC_HEVC_PROFILE_FREXT_GUID,
    };
    const char *szAV1ProfileNames = "main";
    std::vector<GUID> vAV1Profile = std::vector<GUID>{
        NV_ENC_AV1_PROFILE_MAIN_GUID,
    };

    const char *szProfileNames = "(default) auto baseline(h264) main(h264) high(h264) high444(h264)"
        " stereo(h264) progressiv_high(h264) constrained_high(h264)"
        " main(hevc) main10(hevc) frext(hevc)"
        " main(av1) high(av1)";
    std::vector<GUID> vProfile = std::vector<GUID> {
        GUID{},
        NV_ENC_CODEC_PROFILE_AUTOSELECT_GUID,
        NV_ENC_H264_PROFILE_BASELINE_GUID,
        NV_ENC_H264_PROFILE_MAIN_GUID,
        NV_ENC_H264_PROFILE_HIGH_GUID,
        NV_ENC_H264_PROFILE_HIGH_444_GUID,
        NV_ENC_H264_PROFILE_STEREO_GUID,
        NV_ENC_H264_PROFILE_PROGRESSIVE_HIGH_GUID,
        NV_ENC_H264_PROFILE_CONSTRAINED_HIGH_GUID,
        NV_ENC_HEVC_PROFILE_MAIN_GUID,
        NV_ENC_HEVC_PROFILE_MAIN10_GUID,
        NV_ENC_HEVC_PROFILE_FREXT_GUID,
        NV_ENC_AV1_PROFILE_MAIN_GUID,
    };

    const char *szLowLatencyTuningInfoNames = "lowlatency ultralowlatency";
    const char *szTuningInfoNames = "hq lowlatency ultralowlatency lossless";
    std::vector<NV_ENC_TUNING_INFO> vTuningInfo = std::vector<NV_ENC_TUNING_INFO>{
        NV_ENC_TUNING_INFO_HIGH_QUALITY,
        NV_ENC_TUNING_INFO_LOW_LATENCY,
        NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY,
        NV_ENC_TUNING_INFO_LOSSLESS
    };

    const char *szRcModeNames = "constqp vbr cbr";
    std::vector<NV_ENC_PARAMS_RC_MODE> vRcMode = std::vector<NV_ENC_PARAMS_RC_MODE> {
        NV_ENC_PARAMS_RC_CONSTQP,
        NV_ENC_PARAMS_RC_VBR,
        NV_ENC_PARAMS_RC_CBR,
    };

    const char *szMultipass = "disabled qres fullres";
    std::vector<NV_ENC_MULTI_PASS> vMultiPass = std::vector<NV_ENC_MULTI_PASS>{
        NV_ENC_MULTI_PASS_DISABLED,
        NV_ENC_TWO_PASS_QUARTER_RESOLUTION,
        NV_ENC_TWO_PASS_FULL_RESOLUTION,
    };

   const char *szQpMapModeNames = "disabled emphasis_level_map delta_qp_map qp_map";
    std::vector<NV_ENC_QP_MAP_MODE> vQpMapMode = std::vector<NV_ENC_QP_MAP_MODE> {
        NV_ENC_QP_MAP_DISABLED,
        NV_ENC_QP_MAP_EMPHASIS,
        NV_ENC_QP_MAP_DELTA,
        NV_ENC_QP_MAP,
    };


public:
    /*
     * Generates and returns a string describing the values for each field in
     * the NV_ENC_INITIALIZE_PARAMS structure (i.e. a description of the entire
     * set of initialization parameters supplied to the API).
     */
    std::string FullParamToString(const NV_ENC_INITIALIZE_PARAMS *pInitializeParams) {
        std::ostringstream os;
        os << "NV_ENC_INITIALIZE_PARAMS:" << std::endl
            << "encodeGUID: " << ConvertValueToString(vCodec, szCodecNames, pInitializeParams->encodeGUID) << std::endl
            << "presetGUID: " << ConvertValueToString(vPreset, szPresetNames, pInitializeParams->presetGUID) << std::endl;
        if (pInitializeParams->tuningInfo)
        {
            os << "tuningInfo: " << ConvertValueToString(vTuningInfo, szTuningInfoNames, pInitializeParams->tuningInfo) << std::endl;
        }
        os
            << "encodeWidth: " << pInitializeParams->encodeWidth << std::endl
            << "encodeHeight: " << pInitializeParams->encodeHeight << std::endl
            << "darWidth: " << pInitializeParams->darWidth << std::endl
            << "darHeight: " << pInitializeParams->darHeight << std::endl
            << "frameRateNum: " << pInitializeParams->frameRateNum << std::endl
            << "frameRateDen: " << pInitializeParams->frameRateDen << std::endl
            << "enableEncodeAsync: " << pInitializeParams->enableEncodeAsync << std::endl
            << "reportSliceOffsets: " << pInitializeParams->reportSliceOffsets << std::endl
            << "enableSubFrameWrite: " << pInitializeParams->enableSubFrameWrite << std::endl
            << "enableExternalMEHints: " << pInitializeParams->enableExternalMEHints << std::endl
            << "enableMEOnlyMode: " << pInitializeParams->enableMEOnlyMode << std::endl
            << "enableWeightedPrediction: " << pInitializeParams->enableWeightedPrediction << std::endl
            << "maxEncodeWidth: " << pInitializeParams->maxEncodeWidth << std::endl
            << "maxEncodeHeight: " << pInitializeParams->maxEncodeHeight << std::endl
            << "maxMEHintCountsPerBlock: " << pInitializeParams->maxMEHintCountsPerBlock << std::endl
        ;
        NV_ENC_CONFIG *pConfig = pInitializeParams->encodeConfig;
        os << "NV_ENC_CONFIG:" << std::endl
            << "profile: " << ConvertValueToString(vProfile, szProfileNames, pConfig->profileGUID) << std::endl
            << "gopLength: " << pConfig->gopLength << std::endl
            << "frameIntervalP: " << pConfig->frameIntervalP << std::endl
            << "monoChromeEncoding: " << pConfig->monoChromeEncoding << std::endl
            << "frameFieldMode: " << pConfig->frameFieldMode << std::endl
            << "mvPrecision: " << pConfig->mvPrecision << std::endl
            << "NV_ENC_RC_PARAMS:" << std::endl
            << "    rateControlMode: 0x" << std::hex << pConfig->rcParams.rateControlMode << std::dec << std::endl
            << "    constQP: " << pConfig->rcParams.constQP.qpInterP << ", " << pConfig->rcParams.constQP.qpInterB << ", " << pConfig->rcParams.constQP.qpIntra << std::endl
            << "    averageBitRate:  " << pConfig->rcParams.averageBitRate << std::endl
            << "    maxBitRate:      " << pConfig->rcParams.maxBitRate << std::endl
            << "    vbvBufferSize:   " << pConfig->rcParams.vbvBufferSize << std::endl
            << "    vbvInitialDelay: " << pConfig->rcParams.vbvInitialDelay << std::endl
            << "    enableMinQP: " << pConfig->rcParams.enableMinQP << std::endl
            << "    enableMaxQP: " << pConfig->rcParams.enableMaxQP << std::endl
            << "    enableInitialRCQP: " << pConfig->rcParams.enableInitialRCQP << std::endl
            << "    enableAQ: " << pConfig->rcParams.enableAQ << std::endl
            << "    qpMapMode: " << ConvertValueToString(vQpMapMode, szQpMapModeNames, pConfig->rcParams.qpMapMode) << std::endl
            << "    multipass: " << ConvertValueToString(vMultiPass, szMultipass, pConfig->rcParams.multiPass) << std::endl
            << "    enableLookahead: " << pConfig->rcParams.enableLookahead << std::endl
            << "    disableIadapt: " << pConfig->rcParams.disableIadapt << std::endl
            << "    disableBadapt: " << pConfig->rcParams.disableBadapt << std::endl
            << "    enableTemporalAQ: " << pConfig->rcParams.enableTemporalAQ << std::endl
            << "    zeroReorderDelay: " << pConfig->rcParams.zeroReorderDelay << std::endl
            << "    enableNonRefP: " << pConfig->rcParams.enableNonRefP << std::endl
            << "    strictGOPTarget: " << pConfig->rcParams.strictGOPTarget << std::endl
            << "    aqStrength: " << pConfig->rcParams.aqStrength << std::endl
            << "    minQP: " << pConfig->rcParams.minQP.qpInterP << ", " << pConfig->rcParams.minQP.qpInterB << ", " << pConfig->rcParams.minQP.qpIntra << std::endl
            << "    maxQP: " << pConfig->rcParams.maxQP.qpInterP << ", " << pConfig->rcParams.maxQP.qpInterB << ", " << pConfig->rcParams.maxQP.qpIntra << std::endl
            << "    initialRCQP: " << pConfig->rcParams.initialRCQP.qpInterP << ", " << pConfig->rcParams.initialRCQP.qpInterB << ", " << pConfig->rcParams.initialRCQP.qpIntra << std::endl
            << "    temporallayerIdxMask: " << pConfig->rcParams.temporallayerIdxMask << std::endl
            << "    temporalLayerQP: " << (int)pConfig->rcParams.temporalLayerQP[0] << ", " << (int)pConfig->rcParams.temporalLayerQP[1] << ", " << (int)pConfig->rcParams.temporalLayerQP[2] << ", " << (int)pConfig->rcParams.temporalLayerQP[3] << ", " << (int)pConfig->rcParams.temporalLayerQP[4] << ", " << (int)pConfig->rcParams.temporalLayerQP[5] << ", " << (int)pConfig->rcParams.temporalLayerQP[6] << ", " << (int)pConfig->rcParams.temporalLayerQP[7] << std::endl
            << "    targetQuality: " << pConfig->rcParams.targetQuality << std::endl
            << "    lookaheadDepth: " << pConfig->rcParams.lookaheadDepth << std::endl;
        if (pInitializeParams->encodeGUID == NV_ENC_CODEC_H264_GUID) {
            os  
            << "NV_ENC_CODEC_CONFIG (H264):" << std::endl
            << "    enableStereoMVC: " << pConfig->encodeCodecConfig.h264Config.enableStereoMVC << std::endl
            << "    hierarchicalPFrames: " << pConfig->encodeCodecConfig.h264Config.hierarchicalPFrames << std::endl
            << "    hierarchicalBFrames: " << pConfig->encodeCodecConfig.h264Config.hierarchicalBFrames << std::endl
            << "    outputBufferingPeriodSEI: " << pConfig->encodeCodecConfig.h264Config.outputBufferingPeriodSEI << std::endl
            << "    outputPictureTimingSEI: " << pConfig->encodeCodecConfig.h264Config.outputPictureTimingSEI << std::endl
            << "    outputAUD: " << pConfig->encodeCodecConfig.h264Config.outputAUD << std::endl
            << "    disableSPSPPS: " << pConfig->encodeCodecConfig.h264Config.disableSPSPPS << std::endl
            << "    outputFramePackingSEI: " << pConfig->encodeCodecConfig.h264Config.outputFramePackingSEI << std::endl
            << "    outputRecoveryPointSEI: " << pConfig->encodeCodecConfig.h264Config.outputRecoveryPointSEI << std::endl
            << "    enableIntraRefresh: " << pConfig->encodeCodecConfig.h264Config.enableIntraRefresh << std::endl
            << "    enableConstrainedEncoding: " << pConfig->encodeCodecConfig.h264Config.enableConstrainedEncoding << std::endl
            << "    repeatSPSPPS: " << pConfig->encodeCodecConfig.h264Config.repeatSPSPPS << std::endl
            << "    enableVFR: " << pConfig->encodeCodecConfig.h264Config.enableVFR << std::endl
            << "    enableLTR: " << pConfig->encodeCodecConfig.h264Config.enableLTR << std::endl
            << "    qpPrimeYZeroTransformBypassFlag: " << pConfig->encodeCodecConfig.h264Config.qpPrimeYZeroTransformBypassFlag << std::endl
            << "    useConstrainedIntraPred: " << pConfig->encodeCodecConfig.h264Config.useConstrainedIntraPred << std::endl
            << "    level: " << pConfig->encodeCodecConfig.h264Config.level << std::endl
            << "    idrPeriod: " << pConfig->encodeCodecConfig.h264Config.idrPeriod << std::endl
            << "    separateColourPlaneFlag: " << pConfig->encodeCodecConfig.h264Config.separateColourPlaneFlag << std::endl
            << "    disableDeblockingFilterIDC: " << pConfig->encodeCodecConfig.h264Config.disableDeblockingFilterIDC << std::endl
            << "    numTemporalLayers: " << pConfig->encodeCodecConfig.h264Config.numTemporalLayers << std::endl
            << "    spsId: " << pConfig->encodeCodecConfig.h264Config.spsId << std::endl
            << "    ppsId: " << pConfig->encodeCodecConfig.h264Config.ppsId << std::endl
            << "    adaptiveTransformMode: " << pConfig->encodeCodecConfig.h264Config.adaptiveTransformMode << std::endl
            << "    fmoMode: " << pConfig->encodeCodecConfig.h264Config.fmoMode << std::endl
            << "    bdirectMode: " << pConfig->encodeCodecConfig.h264Config.bdirectMode << std::endl
            << "    entropyCodingMode: " << pConfig->encodeCodecConfig.h264Config.entropyCodingMode << std::endl
            << "    stereoMode: " << pConfig->encodeCodecConfig.h264Config.stereoMode << std::endl
            << "    intraRefreshPeriod: " << pConfig->encodeCodecConfig.h264Config.intraRefreshPeriod << std::endl
            << "    intraRefreshCnt: " << pConfig->encodeCodecConfig.h264Config.intraRefreshCnt << std::endl
            << "    maxNumRefFrames: " << pConfig->encodeCodecConfig.h264Config.maxNumRefFrames << std::endl
            << "    sliceMode: " << pConfig->encodeCodecConfig.h264Config.sliceMode << std::endl
            << "    sliceModeData: " << pConfig->encodeCodecConfig.h264Config.sliceModeData << std::endl
            << "    NV_ENC_CONFIG_H264_VUI_PARAMETERS:" << std::endl
            << "        overscanInfoPresentFlag: " << pConfig->encodeCodecConfig.h264Config.h264VUIParameters.overscanInfoPresentFlag << std::endl
            << "        overscanInfo: " << pConfig->encodeCodecConfig.h264Config.h264VUIParameters.overscanInfo << std::endl
            << "        videoSignalTypePresentFlag: " << pConfig->encodeCodecConfig.h264Config.h264VUIParameters.videoSignalTypePresentFlag << std::endl
            << "        videoFormat: " << pConfig->encodeCodecConfig.h264Config.h264VUIParameters.videoFormat << std::endl
            << "        videoFullRangeFlag: " << pConfig->encodeCodecConfig.h264Config.h264VUIParameters.videoFullRangeFlag << std::endl
            << "        colourDescriptionPresentFlag: " << pConfig->encodeCodecConfig.h264Config.h264VUIParameters.colourDescriptionPresentFlag << std::endl
            << "        colourPrimaries: " << pConfig->encodeCodecConfig.h264Config.h264VUIParameters.colourPrimaries << std::endl
            << "        transferCharacteristics: " << pConfig->encodeCodecConfig.h264Config.h264VUIParameters.transferCharacteristics << std::endl
            << "        colourMatrix: " << pConfig->encodeCodecConfig.h264Config.h264VUIParameters.colourMatrix << std::endl
            << "        chromaSampleLocationFlag: " << pConfig->encodeCodecConfig.h264Config.h264VUIParameters.chromaSampleLocationFlag << std::endl
            << "        chromaSampleLocationTop: " << pConfig->encodeCodecConfig.h264Config.h264VUIParameters.chromaSampleLocationTop << std::endl
            << "        chromaSampleLocationBot: " << pConfig->encodeCodecConfig.h264Config.h264VUIParameters.chromaSampleLocationBot << std::endl
            << "        bitstreamRestrictionFlag: " << pConfig->encodeCodecConfig.h264Config.h264VUIParameters.bitstreamRestrictionFlag << std::endl
            << "    ltrNumFrames: " << pConfig->encodeCodecConfig.h264Config.ltrNumFrames << std::endl
            << "    ltrTrustMode: " << pConfig->encodeCodecConfig.h264Config.ltrTrustMode << std::endl
            << "    chromaFormatIDC: " << pConfig->encodeCodecConfig.h264Config.chromaFormatIDC << std::endl
            << "    maxTemporalLayers: " << pConfig->encodeCodecConfig.h264Config.maxTemporalLayers << std::endl;
        } else if (pInitializeParams->encodeGUID == NV_ENC_CODEC_HEVC_GUID) {
            os  
            << "NV_ENC_CODEC_CONFIG (HEVC):" << std::endl
            << "    level: " << pConfig->encodeCodecConfig.hevcConfig.level << std::endl
            << "    tier: " << pConfig->encodeCodecConfig.hevcConfig.tier << std::endl
            << "    minCUSize: " << pConfig->encodeCodecConfig.hevcConfig.minCUSize << std::endl
            << "    maxCUSize: " << pConfig->encodeCodecConfig.hevcConfig.maxCUSize << std::endl
            << "    useConstrainedIntraPred: " << pConfig->encodeCodecConfig.hevcConfig.useConstrainedIntraPred << std::endl
            << "    disableDeblockAcrossSliceBoundary: " << pConfig->encodeCodecConfig.hevcConfig.disableDeblockAcrossSliceBoundary << std::endl
            << "    outputBufferingPeriodSEI: " << pConfig->encodeCodecConfig.hevcConfig.outputBufferingPeriodSEI << std::endl
            << "    outputPictureTimingSEI: " << pConfig->encodeCodecConfig.hevcConfig.outputPictureTimingSEI << std::endl
            << "    outputAUD: " << pConfig->encodeCodecConfig.hevcConfig.outputAUD << std::endl
            << "    enableLTR: " << pConfig->encodeCodecConfig.hevcConfig.enableLTR << std::endl
            << "    disableSPSPPS: " << pConfig->encodeCodecConfig.hevcConfig.disableSPSPPS << std::endl
            << "    repeatSPSPPS: " << pConfig->encodeCodecConfig.hevcConfig.repeatSPSPPS << std::endl
            << "    enableIntraRefresh: " << pConfig->encodeCodecConfig.hevcConfig.enableIntraRefresh << std::endl
            << "    chromaFormatIDC: " << pConfig->encodeCodecConfig.hevcConfig.chromaFormatIDC << std::endl
            << "    pixelBitDepthMinus8: " << pConfig->encodeCodecConfig.hevcConfig.pixelBitDepthMinus8 << std::endl
            << "    idrPeriod: " << pConfig->encodeCodecConfig.hevcConfig.idrPeriod << std::endl
            << "    intraRefreshPeriod: " << pConfig->encodeCodecConfig.hevcConfig.intraRefreshPeriod << std::endl
            << "    intraRefreshCnt: " << pConfig->encodeCodecConfig.hevcConfig.intraRefreshCnt << std::endl
            << "    maxNumRefFramesInDPB: " << pConfig->encodeCodecConfig.hevcConfig.maxNumRefFramesInDPB << std::endl
            << "    ltrNumFrames: " << pConfig->encodeCodecConfig.hevcConfig.ltrNumFrames << std::endl
            << "    vpsId: " << pConfig->encodeCodecConfig.hevcConfig.vpsId << std::endl
            << "    spsId: " << pConfig->encodeCodecConfig.hevcConfig.spsId << std::endl
            << "    ppsId: " << pConfig->encodeCodecConfig.hevcConfig.ppsId << std::endl
            << "    sliceMode: " << pConfig->encodeCodecConfig.hevcConfig.sliceMode << std::endl
            << "    sliceModeData: " << pConfig->encodeCodecConfig.hevcConfig.sliceModeData << std::endl
            << "    maxTemporalLayersMinus1: " << pConfig->encodeCodecConfig.hevcConfig.maxTemporalLayersMinus1 << std::endl
            << "    NV_ENC_CONFIG_HEVC_VUI_PARAMETERS:" << std::endl
            << "        overscanInfoPresentFlag: " << pConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.overscanInfoPresentFlag << std::endl
            << "        overscanInfo: " << pConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.overscanInfo << std::endl
            << "        videoSignalTypePresentFlag: " << pConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.videoSignalTypePresentFlag << std::endl
            << "        videoFormat: " << pConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.videoFormat << std::endl
            << "        videoFullRangeFlag: " << pConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.videoFullRangeFlag << std::endl
            << "        colourDescriptionPresentFlag: " << pConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.colourDescriptionPresentFlag << std::endl
            << "        colourPrimaries: " << pConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.colourPrimaries << std::endl
            << "        transferCharacteristics: " << pConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.transferCharacteristics << std::endl
            << "        colourMatrix: " << pConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.colourMatrix << std::endl
            << "        chromaSampleLocationFlag: " << pConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.chromaSampleLocationFlag << std::endl
            << "        chromaSampleLocationTop: " << pConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.chromaSampleLocationTop << std::endl
            << "        chromaSampleLocationBot: " << pConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.chromaSampleLocationBot << std::endl
            << "        bitstreamRestrictionFlag: " << pConfig->encodeCodecConfig.hevcConfig.hevcVUIParameters.bitstreamRestrictionFlag << std::endl
            << "    ltrTrustMode: " << pConfig->encodeCodecConfig.hevcConfig.ltrTrustMode << std::endl;
        } else if (pInitializeParams->encodeGUID == NV_ENC_CODEC_AV1_GUID) {
            os
                << "NV_ENC_CODEC_CONFIG (AV1):" << std::endl
                << "    level: " << pConfig->encodeCodecConfig.av1Config.level << std::endl
                << "    tier: " << pConfig->encodeCodecConfig.av1Config.tier << std::endl
                << "    minPartSize: " << pConfig->encodeCodecConfig.av1Config.minPartSize << std::endl
                << "    maxPartSize: " << pConfig->encodeCodecConfig.av1Config.maxPartSize << std::endl
                << "    outputAnnexBFormat: " << pConfig->encodeCodecConfig.av1Config.outputAnnexBFormat << std::endl
                << "    enableTimingInfo: " << pConfig->encodeCodecConfig.av1Config.enableTimingInfo << std::endl
                << "    enableDecoderModelInfo: " << pConfig->encodeCodecConfig.av1Config.enableDecoderModelInfo << std::endl
                << "    enableFrameIdNumbers: " << pConfig->encodeCodecConfig.av1Config.enableFrameIdNumbers << std::endl
                << "    disableSeqHdr: " << pConfig->encodeCodecConfig.av1Config.disableSeqHdr << std::endl
                << "    repeatSeqHdr: " << pConfig->encodeCodecConfig.av1Config.repeatSeqHdr << std::endl
                << "    enableIntraRefresh: " << pConfig->encodeCodecConfig.av1Config.enableIntraRefresh << std::endl
                << "    chromaFormatIDC: " << pConfig->encodeCodecConfig.av1Config.chromaFormatIDC << std::endl
                << "    enableBitstreamPadding: " << pConfig->encodeCodecConfig.av1Config.enableBitstreamPadding << std::endl
                << "    enableCustomTileConfig: " << pConfig->encodeCodecConfig.av1Config.enableCustomTileConfig << std::endl
                << "    enableFilmGrainParams: " << pConfig->encodeCodecConfig.av1Config.enableFilmGrainParams << std::endl
                << "    inputPixelBitDepthMinus8: " << pConfig->encodeCodecConfig.av1Config.inputPixelBitDepthMinus8 << std::endl
                << "    pixelBitDepthMinus8: " << pConfig->encodeCodecConfig.av1Config.pixelBitDepthMinus8 << std::endl
                << "    idrPeriod: " << pConfig->encodeCodecConfig.av1Config.idrPeriod << std::endl
                << "    intraRefreshPeriod: " << pConfig->encodeCodecConfig.av1Config.intraRefreshPeriod << std::endl
                << "    intraRefreshCnt: " << pConfig->encodeCodecConfig.av1Config.intraRefreshCnt << std::endl
                << "    maxNumRefFramesInDPB: " << pConfig->encodeCodecConfig.av1Config.maxNumRefFramesInDPB << std::endl
                << "    numTileColumns: " << pConfig->encodeCodecConfig.av1Config.numTileColumns << std::endl
                << "    numTileRows: " << pConfig->encodeCodecConfig.av1Config.numTileRows << std::endl
                << "    maxTemporalLayersMinus1: " << pConfig->encodeCodecConfig.av1Config.maxTemporalLayersMinus1 << std::endl
                << "    colorPrimaries: " << pConfig->encodeCodecConfig.av1Config.colorPrimaries << std::endl
                << "    transferCharacteristics: " << pConfig->encodeCodecConfig.av1Config.transferCharacteristics << std::endl
                << "    matrixCoefficients: " << pConfig->encodeCodecConfig.av1Config.matrixCoefficients << std::endl
                << "    colorRange: " << pConfig->encodeCodecConfig.av1Config.colorRange << std::endl
                << "    chromaSamplePosition: " << pConfig->encodeCodecConfig.av1Config.chromaSamplePosition << std::endl
                << "    useBFramesAsRef: " << pConfig->encodeCodecConfig.av1Config.useBFramesAsRef << std::endl
                << "    numFwdRefs: " << pConfig->encodeCodecConfig.av1Config.numFwdRefs << std::endl
                << "    numBwdRefs: " << pConfig->encodeCodecConfig.av1Config.numBwdRefs << std::endl;
            if (pConfig->encodeCodecConfig.av1Config.filmGrainParams != NULL)
            {
                os
                    << "    NV_ENC_FILM_GRAIN_PARAMS_AV1:" << std::endl
                    << "        applyGrain: " << pConfig->encodeCodecConfig.av1Config.filmGrainParams->applyGrain << std::endl
                    << "        chromaScalingFromLuma: " << pConfig->encodeCodecConfig.av1Config.filmGrainParams->chromaScalingFromLuma << std::endl
                    << "        overlapFlag: " << pConfig->encodeCodecConfig.av1Config.filmGrainParams->overlapFlag << std::endl
                    << "        clipToRestrictedRange: " << pConfig->encodeCodecConfig.av1Config.filmGrainParams->clipToRestrictedRange << std::endl
                    << "        grainScalingMinus8: " << pConfig->encodeCodecConfig.av1Config.filmGrainParams->grainScalingMinus8 << std::endl
                    << "        arCoeffLag: " << pConfig->encodeCodecConfig.av1Config.filmGrainParams->arCoeffLag << std::endl
                    << "        numYPoints: " << pConfig->encodeCodecConfig.av1Config.filmGrainParams->numYPoints << std::endl
                    << "        numCbPoints: " << pConfig->encodeCodecConfig.av1Config.filmGrainParams->numCbPoints << std::endl
                    << "        numCrPoints: " << pConfig->encodeCodecConfig.av1Config.filmGrainParams->numCrPoints << std::endl
                    << "        arCoeffShiftMinus6: " << pConfig->encodeCodecConfig.av1Config.filmGrainParams->arCoeffShiftMinus6 << std::endl
                    << "        grainScaleShift: " << pConfig->encodeCodecConfig.av1Config.filmGrainParams->grainScaleShift << std::endl
                    << "        cbMult: " << pConfig->encodeCodecConfig.av1Config.filmGrainParams->cbMult << std::endl
                    << "        cbLumaMult: " << pConfig->encodeCodecConfig.av1Config.filmGrainParams->cbLumaMult << std::endl
                    << "        cbOffset: " << pConfig->encodeCodecConfig.av1Config.filmGrainParams->cbOffset << std::endl
                    << "        crMult: " << pConfig->encodeCodecConfig.av1Config.filmGrainParams->crMult << std::endl
                    << "        crLumaMult: " << pConfig->encodeCodecConfig.av1Config.filmGrainParams->crLumaMult << std::endl
                    << "        crOffset: " << pConfig->encodeCodecConfig.av1Config.filmGrainParams->crOffset << std::endl;
            }
        }

        return os.str();
    }
};
