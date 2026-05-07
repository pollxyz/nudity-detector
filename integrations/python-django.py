"""
Django integration example. Drop this in your views.py.

Requires:  pip install requests
"""

import logging
import os

import requests
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

log = logging.getLogger(__name__)

MODERATION_URL = os.environ.get("MODERATION_URL", "http://127.0.0.1:8000/check")
MODERATION_TIMEOUT = float(os.environ.get("MODERATION_TIMEOUT", "10"))


def moderate_image(image_file) -> dict:
    """POST the uploaded file to the moderation service. Returns parsed JSON."""
    image_file.seek(0)
    files = {"file": (image_file.name, image_file.read(), image_file.content_type)}
    response = requests.post(MODERATION_URL, files=files, timeout=MODERATION_TIMEOUT)
    response.raise_for_status()
    return response.json()


@csrf_exempt
@require_POST
def upload_profile_picture(request):
    if "avatar" not in request.FILES:
        return JsonResponse({"error": "No file uploaded"}, status=400)

    avatar = request.FILES["avatar"]

    try:
        result = moderate_image(avatar)
    except requests.RequestException as exc:
        # FAIL-CLOSED: reject if moderation can't be reached
        log.exception("Moderation request failed")
        return JsonResponse(
            {"error": "Moderation unavailable, please try again"}, status=503,
        )

    if result["verdict"] == "BLOCK":
        return JsonResponse({
            "error": "Image rejected: inappropriate content",
            "reason": result["reason"],
            "nsfw_score": result["nsfw_score"],
        }, status=400)

    if result["verdict"] == "REVIEW":
        # Save to review queue model, e.g.:
        # ReviewQueue.objects.create(user=request.user, image=avatar, mod_result=result)
        return JsonResponse({"status": "pending_review"})

    # verdict == "ALLOW"
    request.user.profile.avatar = avatar
    request.user.profile.save()
    return JsonResponse({"status": "ok"})


# urls.py:
#   path("api/profile-picture/", upload_profile_picture, name="upload_profile_picture"),
