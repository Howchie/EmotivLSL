# Fallback sampling rate, in Hz.
#
# The EPOC X reader measures the headset's actual rate at startup (it runs at
# either 128 or 256 Hz and the HID report does not say which) and declares the
# measured value on its LSL outlets.  This constant is only the fallback used
# when the rate cannot be measured, and the value to override with
# --sample-rate.  Do not assume it matches a live stream; read
# nominal_srate() from the stream instead.
SRATE = 128
