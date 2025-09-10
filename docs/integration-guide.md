# Integration Guide

This guide explains how to integrate the ExplainGPT-Manim Integration service with your ExplainGPT frontend or other applications.

## Overview

The Manim Integration service provides a REST API that can be easily integrated into any application that needs to generate mathematical animations. It's designed specifically to enhance text-based explanations with visual content.

## Integration Architecture

```
┌─────────────────┐    HTTP/REST    ┌─────────────────┐
│   ExplainGPT    │◄─────────────────►│    Manim        │
│   Frontend      │     JSON         │    Service      │
└─────────────────┘                  └─────────────────┘
        │                                     │
        │                                     │
        ▼                                     ▼
┌─────────────────┐                 ┌─────────────────┐
│   User          │                 │   Generated     │
│   Interface     │                 │   Videos        │
└─────────────────┘                 └─────────────────┘
```

## Basic Integration

### 1. Service Discovery

First, ensure your application can communicate with the Manim service:

```javascript
// Check if service is available
async function checkServiceHealth() {
    try {
        const response = await fetch('http://localhost:8000/healthcheck');
        const health = await response.json();
        return health.status === 'ok';
    } catch (error) {
        console.error('Manim service unavailable:', error);
        return false;
    }
}
```

### 2. Animation Request Wrapper

Create a service class to handle animation requests:

```javascript
class ManimIntegration {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
        this.requestQueue = new Map(); // Track ongoing requests
    }

    /**
     * Generate animation for a given prompt
     * @param {string} prompt - What to animate
     * @param {number} complexity - 1-3 complexity level
     * @param {object} options - Additional options
     * @returns {Promise<string>} Video URL
     */
    async generateAnimation(prompt, complexity = 2, options = {}) {
        // Check if similar request is already in progress
        const requestKey = `${prompt}-${complexity}`;
        if (this.requestQueue.has(requestKey)) {
            return this.requestQueue.get(requestKey);
        }

        const request = this._performRequest(prompt, complexity, options);
        this.requestQueue.set(requestKey, request);

        try {
            const result = await request;
            return result;
        } finally {
            this.requestQueue.delete(requestKey);
        }
    }

    async _performRequest(prompt, complexity, options) {
        // Step 1: Submit job
        const jobResponse = await fetch(`${this.baseUrl}/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt,
                complexity,
                animate: true
            })
        });

        if (!jobResponse.ok) {
            throw new Error(`Failed to submit job: ${jobResponse.statusText}`);
        }

        const { job_id } = await jobResponse.json();

        // Step 2: Poll for completion
        return this._pollForCompletion(job_id, options.timeout || 300000);
    }

    async _pollForCompletion(jobId, timeout) {
        const startTime = Date.now();
        const pollInterval = 2000; // 2 seconds

        while (Date.now() - startTime < timeout) {
            const statusResponse = await fetch(`${this.baseUrl}/status/${jobId}`);
            
            if (!statusResponse.ok) {
                throw new Error(`Failed to get job status: ${statusResponse.statusText}`);
            }

            const status = await statusResponse.json();

            if (status.success && status.video_url) {
                return status.video_url;
            } else if (status.error) {
                throw new Error(`Animation failed: ${status.error}`);
            }

            // Wait before next poll
            await new Promise(resolve => setTimeout(resolve, pollInterval));
        }

        throw new Error('Animation generation timed out');
    }

    /**
     * Get job status without waiting
     * @param {string} jobId 
     * @returns {Promise<object>}
     */
    async getJobStatus(jobId) {
        const response = await fetch(`${this.baseUrl}/status/${jobId}`);
        return response.json();
    }
}
```

## ExplainGPT Frontend Integration

### 1. Explanation Enhancement

Integrate animation generation into your explanation workflow:

```javascript
class EnhancedExplainer {
    constructor() {
        this.manimService = new ManimIntegration();
        this.animationCache = new Map();
    }

    /**
     * Generate explanation with optional animation
     * @param {string} question - User's question
     * @param {boolean} includeAnimation - Whether to generate animation
     * @returns {object} Explanation with text and optional video
     */
    async generateExplanation(question, includeAnimation = true) {
        // Generate text explanation first
        const textExplanation = await this.generateTextExplanation(question);
        
        const result = {
            text: textExplanation,
            animation: null,
            timestamp: Date.now()
        };

        // Determine if animation would be helpful
        if (includeAnimation && this.shouldGenerateAnimation(question, textExplanation)) {
            try {
                const animationPrompt = this.createAnimationPrompt(question, textExplanation);
                const complexity = this.determineComplexity(question);
                
                // Check cache first
                const cacheKey = `${animationPrompt}-${complexity}`;
                if (this.animationCache.has(cacheKey)) {
                    result.animation = this.animationCache.get(cacheKey);
                } else {
                    result.animation = await this.manimService.generateAnimation(
                        animationPrompt, 
                        complexity
                    );
                    
                    // Cache for future use
                    this.animationCache.set(cacheKey, result.animation);
                }
            } catch (error) {
                console.warn('Animation generation failed, continuing with text only:', error);
                // Graceful degradation - explanation still works without animation
            }
        }

        return result;
    }

    shouldGenerateAnimation(question, explanation) {
        // Heuristics to determine if animation would be helpful
        const mathKeywords = [
            'function', 'derivative', 'integral', 'limit', 'graph', 'plot',
            'sine', 'cosine', 'parabola', 'circle', 'triangle', 'vector',
            'matrix', 'transform', 'geometry', 'algebra', 'calculus'
        ];

        const questionLower = question.toLowerCase();
        const explanationLower = explanation.toLowerCase();

        return mathKeywords.some(keyword => 
            questionLower.includes(keyword) || explanationLower.includes(keyword)
        );
    }

    createAnimationPrompt(question, explanation) {
        // Transform the question/explanation into an animation prompt
        // This is where you can add logic to create better prompts
        
        if (question.toLowerCase().includes('derivative')) {
            return `Show how derivatives work as the slope of tangent lines, illustrating the concept from: ${question}`;
        } else if (question.toLowerCase().includes('integral')) {
            return `Visualize integration as the area under a curve, demonstrating: ${question}`;
        } else {
            // Generic approach
            return `Create a mathematical visualization for: ${question}`;
        }
    }

    determineComplexity(question) {
        // Simple heuristic to determine animation complexity
        const complexKeywords = ['proof', 'theorem', 'advanced', 'detailed'];
        const simpleKeywords = ['basic', 'simple', 'introduction'];

        const questionLower = question.toLowerCase();

        if (complexKeywords.some(keyword => questionLower.includes(keyword))) {
            return 3;
        } else if (simpleKeywords.some(keyword => questionLower.includes(keyword))) {
            return 1;
        } else {
            return 2; // Default moderate complexity
        }
    }

    async generateTextExplanation(question) {
        // Your existing text explanation logic
        return "Your text explanation here...";
    }
}
```

### 2. UI Components

#### React Component Example

```jsx
import React, { useState, useEffect } from 'react';

const AnimatedExplanation = ({ question }) => {
    const [explanation, setExplanation] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const explainer = new EnhancedExplainer();

    useEffect(() => {
        const generateExplanation = async () => {
            setLoading(true);
            setError(null);

            try {
                const result = await explainer.generateExplanation(question, true);
                setExplanation(result);
            } catch (err) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        if (question) {
            generateExplanation();
        }
    }, [question]);

    if (loading) {
        return (
            <div className="explanation-loading">
                <div className="spinner"></div>
                <p>Generating explanation and animation...</p>
            </div>
        );
    }

    if (error) {
        return (
            <div className="explanation-error">
                <p>Failed to generate explanation: {error}</p>
            </div>
        );
    }

    return (
        <div className="animated-explanation">
            <div className="text-explanation">
                <p>{explanation.text}</p>
            </div>
            
            {explanation.animation && (
                <div className="animation-container">
                    <video 
                        controls 
                        autoPlay 
                        muted 
                        className="explanation-video"
                        src={explanation.animation}
                    >
                        Your browser does not support video playback.
                    </video>
                </div>
            )}
        </div>
    );
};

export default AnimatedExplanation;
```

#### Vue.js Component Example

```vue
<template>
  <div class="animated-explanation">
    <div v-if="loading" class="loading">
      <div class="spinner"></div>
      <p>Generating explanation...</p>
    </div>
    
    <div v-else-if="error" class="error">
      <p>{{ error }}</p>
    </div>
    
    <div v-else class="explanation-content">
      <div class="text-section">
        <p>{{ explanation.text }}</p>
      </div>
      
      <div v-if="explanation.animation" class="animation-section">
        <video :src="explanation.animation" controls autoplay muted>
          Your browser does not support video playback.
        </video>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'AnimatedExplanation',
  props: {
    question: {
      type: String,
      required: true
    }
  },
  data() {
    return {
      explanation: null,
      loading: false,
      error: null,
      explainer: new EnhancedExplainer()
    }
  },
  watch: {
    question: {
      handler: 'generateExplanation',
      immediate: true
    }
  },
  methods: {
    async generateExplanation() {
      if (!this.question) return;
      
      this.loading = true;
      this.error = null;
      
      try {
        this.explanation = await this.explainer.generateExplanation(
          this.question, 
          true
        );
      } catch (err) {
        this.error = err.message;
      } finally {
        this.loading = false;
      }
    }
  }
}
</script>
```

## Advanced Integration Patterns

### 1. Async Animation Loading

For better user experience, load animations asynchronously:

```javascript
class AsyncAnimationLoader {
    constructor(manimService) {
        this.manimService = manimService;
        this.pendingAnimations = new Map();
    }

    /**
     * Start animation generation in background
     * @param {string} prompt 
     * @param {function} onProgress 
     * @param {function} onComplete 
     */
    startAnimation(prompt, onProgress, onComplete) {
        const jobId = this.generateJobId();
        
        // Start generation
        this.manimService.generateAnimation(prompt, 2)
            .then(videoUrl => {
                onComplete(videoUrl);
                this.pendingAnimations.delete(jobId);
            })
            .catch(error => {
                onComplete(null, error);
                this.pendingAnimations.delete(jobId);
            });

        // Poll progress if needed
        if (onProgress) {
            this.pollProgress(jobId, onProgress);
        }

        return jobId;
    }

    async pollProgress(jobId, onProgress) {
        // Implementation depends on whether service provides progress info
        // For now, just provide status updates
        const statuses = ['Queued', 'Processing', 'Rendering', 'Finalizing'];
        
        for (const status of statuses) {
            onProgress(status);
            await new Promise(resolve => setTimeout(resolve, 5000));
        }
    }

    generateJobId() {
        return Math.random().toString(36).substr(2, 9);
    }
}
```

### 2. Caching Strategy

Implement intelligent caching to improve performance:

```javascript
class AnimationCache {
    constructor(maxSize = 100, ttl = 24 * 60 * 60 * 1000) { // 24 hours
        this.cache = new Map();
        this.maxSize = maxSize;
        this.ttl = ttl;
    }

    set(key, value) {
        // Remove oldest entries if cache is full
        if (this.cache.size >= this.maxSize) {
            const firstKey = this.cache.keys().next().value;
            this.cache.delete(firstKey);
        }

        this.cache.set(key, {
            value,
            timestamp: Date.now()
        });
    }

    get(key) {
        const entry = this.cache.get(key);
        
        if (!entry) return null;
        
        // Check if expired
        if (Date.now() - entry.timestamp > this.ttl) {
            this.cache.delete(key);
            return null;
        }
        
        return entry.value;
    }

    generateKey(prompt, complexity) {
        // Create consistent key for similar requests
        const normalizedPrompt = prompt.toLowerCase().trim();
        return `${normalizedPrompt}-${complexity}`;
    }
}
```

### 3. Error Handling and Fallbacks

Implement robust error handling:

```javascript
class RobustManimIntegration {
    constructor(options = {}) {
        this.primaryService = new ManimIntegration(options.primaryUrl);
        this.fallbackService = options.fallbackUrl ? 
            new ManimIntegration(options.fallbackUrl) : null;
        this.maxRetries = options.maxRetries || 3;
        this.retryDelay = options.retryDelay || 5000;
    }

    async generateAnimation(prompt, complexity = 2) {
        let lastError;
        
        // Try primary service with retries
        for (let i = 0; i < this.maxRetries; i++) {
            try {
                return await this.primaryService.generateAnimation(prompt, complexity);
            } catch (error) {
                lastError = error;
                
                if (i < this.maxRetries - 1) {
                    await new Promise(resolve => setTimeout(resolve, this.retryDelay));
                }
            }
        }

        // Try fallback service if available
        if (this.fallbackService) {
            try {
                return await this.fallbackService.generateAnimation(prompt, complexity);
            } catch (fallbackError) {
                console.warn('Fallback service also failed:', fallbackError);
            }
        }

        throw lastError;
    }
}
```

## Configuration for Production

### Environment Variables

```javascript
// Configuration management for different environments
const config = {
    development: {
        manimServiceUrl: 'http://localhost:8000',
        timeout: 300000, // 5 minutes
        enableCache: true,
        enableAnimations: true
    },
    production: {
        manimServiceUrl: process.env.MANIM_SERVICE_URL || 'https://your-service.com',
        timeout: 600000, // 10 minutes  
        enableCache: true,
        enableAnimations: process.env.ENABLE_ANIMATIONS !== 'false',
        fallbackUrl: process.env.MANIM_FALLBACK_URL
    }
};

const currentConfig = config[process.env.NODE_ENV || 'development'];
```

### Rate Limiting and Quality of Service

```javascript
class QualityOfService {
    constructor(options = {}) {
        this.requestQueue = [];
        this.processing = false;
        this.maxConcurrent = options.maxConcurrent || 3;
        this.currentRequests = 0;
    }

    async queueRequest(requestFn) {
        return new Promise((resolve, reject) => {
            this.requestQueue.push({ requestFn, resolve, reject });
            this.processQueue();
        });
    }

    async processQueue() {
        if (this.processing || this.currentRequests >= this.maxConcurrent) {
            return;
        }

        this.processing = true;

        while (this.requestQueue.length > 0 && this.currentRequests < this.maxConcurrent) {
            const { requestFn, resolve, reject } = this.requestQueue.shift();
            this.currentRequests++;

            requestFn()
                .then(resolve)
                .catch(reject)
                .finally(() => {
                    this.currentRequests--;
                    this.processQueue(); // Process next in queue
                });
        }

        this.processing = false;
    }
}
```

## Testing Integration

### Unit Tests

```javascript
// Jest test example
describe('ManimIntegration', () => {
    let manimService;

    beforeEach(() => {
        manimService = new ManimIntegration('http://test-server');
    });

    test('should generate animation successfully', async () => {
        // Mock fetch responses
        global.fetch = jest.fn()
            .mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ job_id: 'test-job' })
            })
            .mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ 
                    success: true, 
                    video_url: 'http://test.com/video.mp4' 
                })
            });

        const result = await manimService.generateAnimation('test prompt');
        expect(result).toBe('http://test.com/video.mp4');
    });

    test('should handle errors gracefully', async () => {
        global.fetch = jest.fn()
            .mockResolvedValueOnce({
                ok: false,
                statusText: 'Server Error'
            });

        await expect(
            manimService.generateAnimation('test prompt')
        ).rejects.toThrow('Failed to submit job');
    });
});
```

## Performance Monitoring

### Metrics Collection

```javascript
class IntegrationMetrics {
    constructor() {
        this.metrics = {
            requestCount: 0,
            successCount: 0,
            errorCount: 0,
            averageResponseTime: 0,
            cacheHitRate: 0
        };
    }

    recordRequest(duration, success, fromCache = false) {
        this.metrics.requestCount++;
        
        if (success) {
            this.metrics.successCount++;
        } else {
            this.metrics.errorCount++;
        }

        // Update average response time
        this.metrics.averageResponseTime = 
            (this.metrics.averageResponseTime * (this.metrics.requestCount - 1) + duration) / 
            this.metrics.requestCount;

        if (fromCache) {
            this.metrics.cacheHitRate = 
                (this.metrics.cacheHitRate * this.metrics.requestCount + 1) / 
                this.metrics.requestCount;
        }
    }

    getMetrics() {
        return { ...this.metrics };
    }
}
```

This integration guide provides the foundation for successfully incorporating the Manim service into your ExplainGPT frontend or other applications, with robust error handling, caching, and performance optimization.