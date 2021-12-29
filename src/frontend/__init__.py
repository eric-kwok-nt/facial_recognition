import logging
import os
import streamlit.components.v1 as components
from aiortc import RTCPeerConnection, RTCSessionDescription
import streamlit as st

logger = logging.getLogger(___name__)

_RELEASE = False

if not _RELEASE:
    _component_func = components.declare_component(
        "tiny_streamlit_webrtc",
        url="http://localhost:3001",
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component("tiny_streamlit_webrtc", path=build_dir)

def tiny_streamlit_webrtc(key=None):
    component_value = _component_func(key=key, default=0)

    if component_value:
        offer_json = component_value("offerJson")

        # Debug
        st.write(offer_json)
        offer = RTCSessionDescription(sdp=offer["sdp"], type=offer_json["type"])
        pc = RTCPeerConnection()
        
        @pc.on("track")
        def on_track(track):
            """
            Passthrough for server-side implementation with asyncio
            """
            logger.info("Track %s received", track.kind)
            pc.addTrack(track) 
            
            # TODO: Implement video transformation

        # handle offer
        await pc.setRemoteDescription(offer)

        # send answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        # TODO: Send answer back to frontend

    return component_value

if not _RELEASE:
    tiny_streamlit_webrtc()

