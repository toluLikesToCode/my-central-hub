/* eslint-disable @typescript-eslint/no-unused-vars */
/* eslint-disable @typescript-eslint/ban-ts-comment */
// @ts-nocheck

import { IncomingRequest, SecurityContext } from '../../src/entities/http';

describe('IncomingRequest with SecurityContext', () => {
  test('should create a valid request with security context', () => {
    const securityContext: SecurityContext = {
      authenticated: true,
      clientIp: '192.168.1.1',
      validatedAt: new Date(),
      origin: 'https://trusted-site.com',
      token: 'jwt-token-123',
      flags: {
        isAdmin: true,
        hasPremiumAccess: true,
      },
      rateLimit: {
        remaining: 95,
        limit: 100,
        reset: Date.now() + 3600000,
      },
    };

    const req: IncomingRequest = {
      url: new URL('http://localhost:8080/api/data'),
      path: '/api/data',
      query: { filter: 'active' },
      httpVersion: 'HTTP/1.1',
      method: 'GET',
      headers: {
        'content-type': 'application/json',
        authorization: 'Bearer token123',
      },
      raw: 'GET /api/data HTTP/1.1\r\nContent-Type: application/json\r\n\r\n',
      security: securityContext,
      timing: {
        startedAt: Date.now(),
        parsedAt: Date.now() + 5,
        routedAt: Date.now() + 10,
      },
    };

    // Test that the structure matches our interface
    expect(req.security).toBeDefined();
    expect(req.security?.authenticated).toBe(true);
    expect(req.security?.clientIp).toBe('192.168.1.1');
    expect(req.security?.validatedAt).toBeInstanceOf(Date);
    expect(req.security?.token).toBe('jwt-token-123');
    expect(req.security?.flags?.isAdmin).toBe(true);
    expect(req.security?.rateLimit?.remaining).toBe(95);
    expect(req.security?.rateLimit?.limit).toBe(100);

    // Test timing data
    expect(req.timing).toBeDefined();
    expect(req.timing?.startedAt).toBeDefined();
    expect(req.timing?.parsedAt).toBeGreaterThan(req.timing?.startedAt);
  });

  test('should maintain backward compatibility with legacy requests', () => {
    // Create a request without the new security fields
    const legacyReq: IncomingRequest = {
      url: new URL('http://localhost:8080/api/legacy'),
      path: '/api/legacy',
      query: {},
      httpVersion: 'HTTP/1.1',
      method: 'GET',
      headers: {},
      raw: 'GET /api/legacy HTTP/1.1\r\n\r\n',
    };

    // Should not throw errors when accessing undefined security properties
    expect(() => {
      const authenticated = legacyReq.security?.authenticated;
      const clientIp = legacyReq.security?.clientIp;
    }).not.toThrow();

    // All security properties should be undefined but accessible
    expect(legacyReq.security).toBeUndefined();
    expect(legacyReq.timing).toBeUndefined();
    expect(legacyReq.validationErrors).toBeUndefined();
  });

  test('should support validation errors', () => {
    const reqWithErrors: IncomingRequest = {
      url: new URL('http://localhost:8080/api/data'),
      path: '/api/data',
      query: {},
      httpVersion: 'HTTP/1.1',
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: Buffer.from('{"invalid": "json"}'),
      raw: 'POST /api/data HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{"invalid": "json"}',
      validationErrors: ['Missing required field: name', 'Invalid email format'],
    };

    expect(reqWithErrors.validationErrors).toHaveLength(2);
    expect(reqWithErrors.validationErrors![0]).toBe('Missing required field: name');
    expect(reqWithErrors.validationErrors![1]).toBe('Invalid email format');
  });
});
