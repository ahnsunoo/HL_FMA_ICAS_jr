#ifndef _ROS_SERVICE_all_data_reset_h
#define _ROS_SERVICE_all_data_reset_h
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "ros/msg.h"

namespace iahrs_driver
{

static const char ALL_DATA_RESET[] = "iahrs_driver/all_data_reset";

  class all_data_resetRequest : public ros::Msg
  {
    public:

    all_data_resetRequest()
    {
    }

    virtual int serialize(unsigned char *outbuffer) const override
    {
      int offset = 0;
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer) override
    {
      int offset = 0;
     return offset;
    }

    virtual const char * getType() override { return ALL_DATA_RESET; };
    virtual const char * getMD5() override { return "d41d8cd98f00b204e9800998ecf8427e"; };

  };

  class all_data_resetResponse : public ros::Msg
  {
    public:
      typedef bool _command_Result_type;
      _command_Result_type command_Result;

    all_data_resetResponse():
      command_Result(0)
    {
    }

    virtual int serialize(unsigned char *outbuffer) const override
    {
      int offset = 0;
      union {
        bool real;
        uint8_t base;
      } u_command_Result;
      u_command_Result.real = this->command_Result;
      *(outbuffer + offset + 0) = (u_command_Result.base >> (8 * 0)) & 0xFF;
      offset += sizeof(this->command_Result);
      return offset;
    }

    virtual int deserialize(unsigned char *inbuffer) override
    {
      int offset = 0;
      union {
        bool real;
        uint8_t base;
      } u_command_Result;
      u_command_Result.base = 0;
      u_command_Result.base |= ((uint8_t) (*(inbuffer + offset + 0))) << (8 * 0);
      this->command_Result = u_command_Result.real;
      offset += sizeof(this->command_Result);
     return offset;
    }

    virtual const char * getType() override { return ALL_DATA_RESET; };
    virtual const char * getMD5() override { return "6b370111df6d9d8116fbdd43bd394a52"; };

  };

  class all_data_reset {
    public:
    typedef all_data_resetRequest Request;
    typedef all_data_resetResponse Response;
  };

}
#endif
