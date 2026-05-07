<?php
/**
 * Laravel/PHP integration example.
 *
 * Drop this in app/Http/Controllers/ProfilePictureController.php
 *
 * Requires:  Guzzle (already in Laravel)
 *            composer require guzzlehttp/guzzle  # if not present
 */

namespace App\Http\Controllers;

use GuzzleHttp\Client;
use GuzzleHttp\Exception\GuzzleException;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Log;

class ProfilePictureController extends Controller
{
    private const MODERATION_URL = 'http://127.0.0.1:8000/check';
    private const TIMEOUT_SECS = 10;

    public function upload(Request $request)
    {
        $request->validate([
            'avatar' => 'required|image|max:10240', // 10 MB
        ]);

        $file = $request->file('avatar');

        try {
            $result = $this->moderate($file->getPathname(), $file->getMimeType());
        } catch (GuzzleException $e) {
            // FAIL-CLOSED: reject if moderation service is unreachable
            Log::error('Moderation service unavailable', ['error' => $e->getMessage()]);
            return response()->json([
                'error' => 'Moderation unavailable, please try again',
            ], 503);
        }

        switch ($result['verdict']) {
            case 'BLOCK':
                return response()->json([
                    'error' => 'Image rejected: inappropriate content',
                    'reason' => $result['reason'],
                    'nsfw_score' => $result['nsfw_score'],
                ], 400);

            case 'REVIEW':
                // Save to review queue
                // ReviewQueue::create([...]);
                return response()->json(['status' => 'pending_review']);

            case 'ALLOW':
            default:
                $path = $file->store('avatars', 'public');
                $request->user()->update(['avatar_path' => $path]);
                return response()->json(['status' => 'ok']);
        }
    }

    private function moderate(string $filePath, string $mimeType): array
    {
        $client = new Client(['timeout' => self::TIMEOUT_SECS]);
        $response = $client->post(self::MODERATION_URL, [
            'multipart' => [
                [
                    'name' => 'file',
                    'contents' => fopen($filePath, 'r'),
                    'headers' => ['Content-Type' => $mimeType],
                ],
            ],
        ]);
        return json_decode((string) $response->getBody(), true);
    }
}

// routes/web.php (or api.php):
//   Route::post('/api/profile-picture', [ProfilePictureController::class, 'upload']);
