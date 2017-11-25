/*
 * Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifdef USE_CSI_OV10640

#if __STDC_VERSION__ >= 199901L
#define _XOPEN_SOURCE 600
#else
#define _XOPEN_SOURCE 500
#endif /* __STDC_VERSION__ */

#include <string.h>
#include <stdlib.h>

#include "plugin.h"
#include "log_utils.h"

#define PLUGIN_STATEMENT_BUFFER_SIZE        16384

// Get the size of a one-dimensional array
#define ARRAY_SIZE(a)       (sizeof(a) / sizeof((a)[0]))

// get the number of columns in a matrix
#define MATRIX_COLUMNS(m)       ARRAY_SIZE(m[0])

typedef enum {
    PluginConfigIdAWB_Enable,
    PluginConfigIdAWB_Matrix,
    PluginConfigIdAWB_gains,
    PluginConfigIdAWB_Threshholds,
    PluginConfigIdAWB_Points
} PluginConfigIdAWB;

typedef enum {
    PluginConfigIdAE_Enable
} PluginConfigIdAE;

typedef struct {
    unsigned int id;
    const char *name;
} PluginConfigId;

typedef NvMediaStatus
(*ParseConfigStatementFunc)(
    const char* statement,
    PluginConfigData *data);

typedef struct {
    const char      *name;
    ParseConfigStatementFunc parse;
} PluginConfigParser;

static NvMediaStatus
IPPPluginParserParseStatement(
    const char* statement,
    PluginConfigData *data);

static NvMediaStatus
IPPPluginParserArrayIndex(const char *statement, unsigned int *val, int *skip);

static NvMediaStatus
IPPPluginParserFloat(const char *statement, float *val);

static NvMediaStatus
IPPPluginParserUint(const char *statement, unsigned int *val);

static NvMediaStatus
IPPPluginParserInt(const char *statement, int *val);

static NvMediaStatus
IPPPluginParserBool(const char *statement, NvMediaBool *value);

static NvMediaStatus
IPPPluginParserUintArray(const char *statement, int count, unsigned int *vals);

static NvMediaStatus
IPPPluginParserIntArray(const char *statement, int count, int *vals);

static NvMediaStatus
IPPPluginParserFloatArray(const char *statement, int count, float *vals);

static NvMediaStatus
IPPPluginParserAwb(
    const char* statement,
    PluginConfigData *data);

static NvMediaStatus
IPPPluginParserAe(
    const char* statement,
    PluginConfigData *data);

NvMediaStatus
IPPPluginParserArrayIndex(const char *statement, unsigned int *val, int *skip)
{
    char tmp[32], *c = tmp;
    int  i = 0;

    if (statement[0] != '[') {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }
    statement++;
    while (i < 31 && *statement != ']' && *statement != '\0') {
        *c++ = *statement++;
        i++;
    }
    if (*statement != ']') {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }
    *c = '\0';
    *skip = i + 2;
    return IPPPluginParserUint(tmp, val);
}


NvMediaStatus
IPPPluginParserFloat(const char *statement, float *val)
{
    *val = (float)strtod(statement, NULL);
    return NVMEDIA_STATUS_OK;
}

NvMediaStatus
IPPPluginParserUint(const char *statement, unsigned int *val)
{
    unsigned int value = 0;

    while (*statement == '0') {
        statement++;
    }

    if (*statement == 'x' || *statement == 'X') {
        statement++;
        while (*statement != '\0' &&
               (((*statement >= '0') && (*statement <= '9')) ||
                ((*statement >= 'a') && (*statement <= 'f')) ||
                ((*statement >= 'A') && (*statement <= 'F')))) {
            value *= 16;
            if ((*statement >= '0') && (*statement <= '9')) {
                value += (*statement - '0');
            }
            else if ((*statement >= 'A') && (*statement <= 'F')) {
                value += ((*statement - 'A')+10);
            }
            else {
                value += ((*statement - 'a')+10);
            }
            statement++;
        }
        if (*statement != '\0') {
            *val = 0;
            return NVMEDIA_STATUS_BAD_PARAMETER;
        }
    }
    else {
        while (*statement != '\0' &&
               *statement >= '0' &&
               *statement <= '9') {
            value = value*10 + (*statement - '0');
            statement++;
        }
        if (*statement != '\0') {
            *val = 0;
            return NVMEDIA_STATUS_BAD_PARAMETER;
        }
    }
    *val = value;
    return NVMEDIA_STATUS_OK;
}

NvMediaStatus
IPPPluginParserInt(const char *statement, int *val)
{
    NvMediaBool isNeg = (statement[0] == '-');
    unsigned int  value = 0;

    if (isNeg) {
        statement++;
    }

    if (IPPPluginParserUint(statement, &value) != NVMEDIA_STATUS_OK) {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    *val = (isNeg) ? -((int)value) : (int)value;
    return NVMEDIA_STATUS_OK;
}

NvMediaStatus
IPPPluginParserBool(const char *statement, NvMediaBool *value)
{
    int v;

    if (strcmp(statement, "TRUE") == 0 ||
        strcmp(statement, "true") == 0) {
        *value = NVMEDIA_TRUE;
    }

    else if (strcmp(statement, "FALSE") == 0 ||
             strcmp(statement, "false") == 0) {
        *value = NVMEDIA_FALSE;
    }
    else if (IPPPluginParserInt(statement, &v) == NVMEDIA_STATUS_OK) {
        *value = (v != 0);
    }
    else {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }

    return NVMEDIA_STATUS_OK;
}

NvMediaStatus
IPPPluginParserUintArray(const char *statement, int count, unsigned int *vals)
{
    char tmp[32], *c;
    int i, j = 0;

    if (statement[0] != '{') {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }
    statement++;

    while (*statement != '\0' && *statement != '}' && j != count) {
        i = 0;
        c = tmp;
        while (i < 31 && *statement != '}' &&
               *statement != ',' && *statement != '\0') {
            *c++ = *statement++;
            i++;
        }
        if (i >= 31) {
            return NVMEDIA_STATUS_BAD_PARAMETER;
        }
        else if (*statement == ',') {
            statement++;
        }
        *c = '\0';
        if (IPPPluginParserUint(tmp, &vals[j++]) != NVMEDIA_STATUS_OK) {
            return NVMEDIA_STATUS_BAD_PARAMETER;
        }
    }

    return (j == count && *statement == '}')
                ? NVMEDIA_STATUS_OK
                : NVMEDIA_STATUS_BAD_PARAMETER;
}

NvMediaStatus
IPPPluginParserIntArray(const char *statement, int count, int *vals)
{
    char tmp[32], *c;
    int i, j = 0;

    if (statement[0] != '{') {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }
    statement++;

    while (*statement != '\0' && *statement != '}' && j != count) {
        i = 0;
        c = tmp;
        while (i < 31 && *statement != '}' &&
               *statement != ',' && *statement != '\0') {
            *c++ = *statement++;
            i++;
        }
        if (i >= 31) {
            return NVMEDIA_STATUS_BAD_PARAMETER;
        }
        else if (*statement == ',') {
            statement++;
        }
        *c = '\0';
        if (IPPPluginParserInt(tmp, &vals[j++]) != NVMEDIA_STATUS_OK) {
            return NVMEDIA_STATUS_BAD_PARAMETER;
        }
    }

    return (j == count && *statement == '}')
                ? NVMEDIA_STATUS_OK
                : NVMEDIA_STATUS_BAD_PARAMETER;
}

NvMediaStatus
IPPPluginParserFloatArray(const char *statement, int count, float *vals)
{
    char tmp[32], *c;
    int i, j = 0;

    if (statement[0] != '{') {
        return NVMEDIA_STATUS_BAD_PARAMETER;
    }
    statement++;

    while (*statement != '\0' && *statement != '}' && j != count) {
        i = 0;
        c = tmp;
        while (i < 31 && *statement != '}' && *statement != ','
               && *statement != '\0') {
            *c++ = *statement++;
            i++;
        }
        if (i >= 31) {
            return NVMEDIA_STATUS_BAD_PARAMETER;
        }
        else if (*statement == ',') {
            statement++;
        }
        *c = '\0';
        if (IPPPluginParserFloat(tmp, &vals[j++]) != NVMEDIA_STATUS_OK) {
            return NVMEDIA_STATUS_BAD_PARAMETER;
        }
    }

    return (j == count && *statement == '}')
                ? NVMEDIA_STATUS_OK
                : NVMEDIA_STATUS_BAD_PARAMETER;
}

/** Finds a parser whose name matches the statement
 * \param[in] statement A configuration statement.
 * \param[in] parsers An array of PluginConfigParser.
 * \param[in] length The length of the array of parsers.
 * \return int. Return the index of the found parser
 *  or -1 if the parser is not found.
 */
static int
IPPPluginParserFind (
    const char* statement,
    PluginConfigParser *parsers,
    int length)
{
    int i = 0;
    for(i = 0; i < length; i++) {
        if( strncmp(
                parsers[i].name,
                statement,
                strlen(parsers[i].name) ) == 0)
        {
            break;
        }
    }
    return (i < length) ? i : -1;
}

NvMediaStatus
IPPPluginParserAwb(
    const char* statement,
    PluginConfigData *data)
{
    PluginConfigId awbConfigs[] = {
        { PluginConfigIdAWB_Enable, "enable=" },
        // For a matrix, the equality sign will be skipped
        // after calling IPPPluginParser_arrayIndex().
        { PluginConfigIdAWB_Matrix, "matrix" },
        { PluginConfigIdAWB_gains, "gains=" },
        { PluginConfigIdAWB_Threshholds, "threshholds=" },
        { PluginConfigIdAWB_Points, "points=" }
    };
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    int skip;
    unsigned int i, index;

    for(i = 0; i < ARRAY_SIZE(awbConfigs); i++) {
        if( strncmp(
                awbConfigs[i].name,
                statement,
                strlen(awbConfigs[i].name) ) != 0 ) {
            continue;
        }

        statement += strlen(awbConfigs[i].name);

        switch(awbConfigs[i].id) {
        case PluginConfigIdAWB_Enable:
            status = IPPPluginParserBool(statement, &data->awb.enable);
            break;

        case PluginConfigIdAWB_Matrix:
            status = IPPPluginParserArrayIndex(statement, &index, &skip);
            if(status != NVMEDIA_STATUS_OK) break;

            if(index > MATRIX_COLUMNS(data->awb.matrix)) {
                status = NVMEDIA_STATUS_BAD_PARAMETER;
                break;
            }
            statement += skip;
            if(*statement != '=') {
                status = NVMEDIA_STATUS_BAD_PARAMETER;
                break;
            }
            ++statement;

            status = IPPPluginParserFloatArray(
                        statement,
                        MATRIX_COLUMNS(data->awb.matrix),
                        data->awb.matrix[index]);
            break;

        case PluginConfigIdAWB_gains:
            status = IPPPluginParserFloatArray(
                        statement,
                        ARRAY_SIZE(data->awb.gains),
                        data->awb.gains);
            break;

        case PluginConfigIdAWB_Threshholds:
            status = IPPPluginParserUintArray(
                        statement,
                        ARRAY_SIZE(data->awb.threshholds),
                        data->awb.threshholds);
            break;

        case PluginConfigIdAWB_Points:
            status = IPPPluginParserIntArray(
                        statement,
                        ARRAY_SIZE(data->awb.points),
                        data->awb.points);
            break;
        }
    }
    return status;
}

NvMediaStatus
IPPPluginParserAe(
    const char* statement,
    PluginConfigData *data)
{
    return NVMEDIA_STATUS_OK;
}

NvMediaStatus
IPPPluginParserParseStatement(
    const char* statement,
    PluginConfigData *data)
{
    PluginConfigParser parser;
    int i;

    //  all top-level names include the trailing "."
    PluginConfigParser parsers[] = {
        { "awb.", IPPPluginParserAwb },
        { "ae.", IPPPluginParserAe }
    };


    i = IPPPluginParserFind (
            statement,
            parsers,
            ARRAY_SIZE(parsers));

    if (i < 0) return NVMEDIA_STATUS_OK;

    parser = parsers[i];

    statement += strlen(parser.name);

    return parser.parse(statement, data);

}

NvMediaStatus
IPPPluginParseConfiguration(
    NvMediaIPPPlugin *pluginHandle,
    const char *configString)
{
    char buffer[PLUGIN_STATEMENT_BUFFER_SIZE];
    int  space = 0;
    char current;
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    NvMediaBool isComment = NVMEDIA_FALSE;
    NvMediaBool equalEncountered = NVMEDIA_FALSE;
    unsigned int index;
    unsigned int lineCount = 1;
    unsigned int configLength;
    unsigned int i;
    PluginContext *ctx = (PluginContext *)pluginHandle;

    if(!configString) {
        return NVMEDIA_STATUS_OK;
    }

    if(!pluginHandle) {
        LOG_ERR("%s: No plugin context passed\n", __func__);
        return NVMEDIA_STATUS_ERROR;
    }

    configLength = strlen(configString);

    for(index = 0; index < configLength; index++) {
        current = configString[index];

        if (space >= (PLUGIN_STATEMENT_BUFFER_SIZE - 1)) {
            // if we exceeded the max buffer size, it is likely
            // due to a missing semi-colon at the end of a line
            status = NVMEDIA_STATUS_INSUFFICIENT_BUFFERING;
            LOG_ERR("%s: insufficient buffering\n", __func__);
            goto done;
        }

        switch (current) {
        case ';':
            if (!isComment)
            {
                buffer[space++] = '\0';

                status = IPPPluginParserParseStatement(buffer, &ctx->configs);

                if(status == NVMEDIA_STATUS_OK)
                {
                    space = 0;
                    equalEncountered = NVMEDIA_FALSE;
                    continue;
                }

                if (status != NVMEDIA_STATUS_OK) {
                    LOG_DBG("%s: Error parsing: %s\n", __func__,  buffer);
                    // In case an unrecognized parameter is seen, allow it to
                    // get reported above, but change status to prevent the
                    // driver from bailing.
                    status = NVMEDIA_STATUS_OK;
                }
                space = 0;
                equalEncountered = NVMEDIA_FALSE;
            }
            break;

        //  ignore whitespaces
        case '\n':
            lineCount++;
            // fall through
        case '\r':
            // carriage returns end comments
            isComment = NVMEDIA_FALSE;
        case ' ':
        case '\t':
            break;

        case '#':
            isComment = NVMEDIA_TRUE;
            break;

        default:
            if(isComment) {
                continue;
            }

            buffer[space++] = current;
            if (current == '=') {
                if (!equalEncountered) {
                    equalEncountered = NVMEDIA_TRUE;
                } else {
                    LOG_ERR("%s: syntax error: two equality signs encountered at line %d\n",
                        __func__, lineCount);
                    status = NVMEDIA_STATUS_BAD_PARAMETER;
                    goto done;
                }
            }
            break;
        } // switch
    } // for

done:
    if(status != NVMEDIA_STATUS_OK) {
        LOG_ERR("%s: @line #%d error code %#x\n", __func__, lineCount, status);
    }

    for(i = 0; i < ARRAY_SIZE(ctx->configs.awb.gains); i++) {
        LOG_INFO("%s: awb.gains[%d] = %f\n",
            __func__, i, ctx->configs.awb.gains[i]);
    }

    for(i = 0; i < ARRAY_SIZE(ctx->configs.awb.points); i++) {
        LOG_INFO("%s: awb.points[%d] = %d\n",
            __func__, i, ctx->configs.awb.points[i]);
    }

    return status;
}

#endif // USE_CSI_OV10640
