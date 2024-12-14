namespace converter
{
    void RGBtoHSV(float& fR, float& fG, float fB, float& fH, float& fS, float& fV) {
        // Copyright (c) 2014, Jan Winkler <winkler@cs.uni-bremen.de>
        // All rights reserved.
        //
        // Redistribution and use in source and binary forms, with or without
        // modification, are permitted provided that the following conditions are met:
        //
        //     * Redistributions of source code must retain the above copyright
        //       notice, this list of conditions and the following disclaimer.
        //     * Redistributions in binary form must reproduce the above copyright
        //       notice, this list of conditions and the following disclaimer in the
        //       documentation and/or other materials provided with the distribution.
        //     * Neither the name of Universität Bremen nor the names of its
        //       contributors may be used to endorse or promote products derived from
        //       this software without specific prior written permission.
        //
        // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
        // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
        // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
        // ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
        // LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
        // CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
        // SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
        // INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
        // CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
        // ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
        // POSSIBILITY OF SUCH DAMAGE.

        /* Author: Jan Winkler */

        //  https://gist.github.com/fairlight1337/4935ae72bcbcc1ba5c72

        float fCMax = max(max(fR, fG), fB);
        float fCMin = min(min(fR, fG), fB);
        float fDelta = fCMax - fCMin;

        if (fDelta > 0) {
            if (fCMax == fR) {
                fH = 60 * (fmod(((fG - fB) / fDelta), 6));
            }
            else if (fCMax == fG) {
                fH = 60 * (((fB - fR) / fDelta) + 2);
            }
            else if (fCMax == fB) {
                fH = 60 * (((fR - fG) / fDelta) + 4);
            }

            if (fCMax > 0) {
                fS = fDelta / fCMax;
            }
            else {
                fS = 0;
            }

            fV = fCMax;
        }
        else {
            fH = 0;
            fS = 0;
            fV = fCMax;
        }

        if (fH < 0) {
            fH = 360 + fH;
        }
    }
}