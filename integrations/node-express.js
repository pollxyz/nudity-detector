/**
 * Node.js / Express integration example.
 *
 * Drop this in your upload handler. Posts the uploaded image bytes to the
 * moderation service and acts on the verdict.
 *
 * Requires:  npm install multer node-fetch form-data
 *            (or in modern Node: native fetch + FormData)
 */

const express = require('express');
const multer = require('multer');
const FormData = require('form-data');

// On Node 18+ fetch is global; otherwise: const fetch = require('node-fetch');

const MODERATION_URL = process.env.MODERATION_URL || 'http://127.0.0.1:8000/check';
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 10 * 1024 * 1024 } });
const app = express();

async function moderateImage(buffer, filename, mimetype) {
    const form = new FormData();
    form.append('file', buffer, { filename, contentType: mimetype });
    const response = await fetch(MODERATION_URL, {
        method: 'POST',
        body: form,
        headers: form.getHeaders ? form.getHeaders() : undefined,
    });
    if (!response.ok) {
        throw new Error(`Moderation service returned ${response.status}`);
    }
    return response.json();
}

app.post('/api/profile-picture', upload.single('avatar'), async (req, res) => {
    if (!req.file) return res.status(400).json({ error: 'No file uploaded' });

    let result;
    try {
        result = await moderateImage(req.file.buffer, req.file.originalname, req.file.mimetype);
    } catch (err) {
        // FAIL-CLOSED: if the moderation service is down, do NOT accept the image.
        // Change to fail-open by replacing this with `result = { verdict: 'ALLOW' }` if you prefer.
        console.error('Moderation failed:', err);
        return res.status(503).json({ error: 'Moderation unavailable, please try again' });
    }

    switch (result.verdict) {
        case 'BLOCK':
            return res.status(400).json({
                error: 'Image rejected: inappropriate content',
                reason: result.reason,
                nsfw_score: result.nsfw_score,
            });
        case 'REVIEW':
            // Save image but mark for human moderation queue
            await saveToReviewQueue(req.user.id, req.file.buffer, result);
            return res.json({ status: 'pending_review' });
        case 'ALLOW':
        default:
            await savePofileImage(req.user.id, req.file.buffer);
            return res.json({ status: 'ok' });
    }
});

async function saveToReviewQueue(userId, imageBuffer, modResult) { /* your code */ }
async function savePofileImage(userId, imageBuffer) { /* your code */ }

app.listen(3000);
