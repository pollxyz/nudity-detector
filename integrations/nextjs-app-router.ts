/**
 * Next.js (App Router) integration example.
 *
 * File: app/api/profile-picture/route.ts
 *
 * Posts the uploaded image to the moderation service from the server side
 * (never from the browser — keeps the moderation URL private).
 */

import { NextRequest, NextResponse } from 'next/server';

const MODERATION_URL = process.env.MODERATION_URL || 'http://127.0.0.1:8000/check';

interface ModerationResult {
    verdict: 'ALLOW' | 'REVIEW' | 'BLOCK';
    reason: string;
    nsfw_score: number;
    safe_score: number;
    explicit_detections: Array<{
        class: string;
        category: string;
        score: number;
        box: number[];
    }>;
    elapsed_ms: number;
}

async function moderateImage(file: File): Promise<ModerationResult> {
    const form = new FormData();
    form.append('file', file);
    const response = await fetch(MODERATION_URL, {
        method: 'POST',
        body: form,
    });
    if (!response.ok) {
        throw new Error(`Moderation service returned ${response.status}`);
    }
    return response.json();
}

export async function POST(req: NextRequest) {
    const formData = await req.formData();
    const file = formData.get('avatar') as File | null;
    if (!file) {
        return NextResponse.json({ error: 'No file uploaded' }, { status: 400 });
    }

    let result: ModerationResult;
    try {
        result = await moderateImage(file);
    } catch (err) {
        // FAIL-CLOSED: reject if moderation is down
        console.error('Moderation failed:', err);
        return NextResponse.json(
            { error: 'Moderation unavailable, please try again' },
            { status: 503 },
        );
    }

    switch (result.verdict) {
        case 'BLOCK':
            return NextResponse.json(
                {
                    error: 'Image rejected: inappropriate content',
                    reason: result.reason,
                    nsfw_score: result.nsfw_score,
                },
                { status: 400 },
            );
        case 'REVIEW':
            // await db.reviewQueue.create({ ... });
            return NextResponse.json({ status: 'pending_review' });
        case 'ALLOW':
        default:
            // await uploadToS3(...);
            return NextResponse.json({ status: 'ok' });
    }
}
