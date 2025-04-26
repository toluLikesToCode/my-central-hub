import { getMimeType } from '../../src/utils/helpers';

describe('Helpers - getMimeType', () => {
  it('should return correct MIME type for mp4', () => {
    expect(getMimeType('movie.mp4')).toBe('video/mp4');
  });

  it('should return correct MIME type for jpg', () => {
    expect(getMimeType('image.jpg')).toBe('image/jpeg');
  });

  it('should return default MIME type for unknown file extension', () => {
    expect(getMimeType('something.unknownext')).toBe('application/octet-stream');
  });

  it('should return default MIME type for file with no extension', () => {
    expect(getMimeType('README')).toBe('application/octet-stream');
  });
});
