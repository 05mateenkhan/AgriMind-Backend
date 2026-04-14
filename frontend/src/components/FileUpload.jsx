import { useRef } from 'react';

const FileUpload = ({ onChange, preview, isUploading }) => {
  const inputRef = useRef(null);

  const handleClick = () => inputRef.current?.click();

  return (
    <div className="space-y-4">
      <button
        type="button"
        onClick={handleClick}
        className="w-full rounded-2xl border border-dashed border-border-cream px-6 py-10 flex flex-col items-center justify-center gap-3 bg-cream-card hover:bg-cream-highlight transition text-center"
      >
        <p className="text-lg font-medium text-lime-800">Upload Plant Image</p>
        <p className="text-text-secondary text-sm">
          PNG, JPG up to 5MB. Drag & drop supported.
        </p>
      </button>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={onChange}
        disabled={isUploading}
      />
      {preview && (
        <div className="rounded-2xl overflow-hidden border border-border-cream bg-cream-card">
          <img src={preview} alt="preview" className="w-full object-cover" />
        </div>
      )}
    </div>
  );
};

export default FileUpload;

