export default function SkeletonCard() {
  return (
    <div className="glass-card flex flex-col gap-3 p-5">
      <div className="flex items-start gap-3">
        <div className="skeleton h-12 w-12 flex-shrink-0 rounded-2xl" />
        <div className="flex flex-1 flex-col gap-2">
          <div className="skeleton h-4 w-3/4 rounded" />
          <div className="skeleton h-3 w-1/3 rounded" />
        </div>
      </div>
      <div className="skeleton h-3 w-full rounded" />
      <div className="skeleton h-3 w-5/6 rounded" />
      <div className="skeleton h-3 w-4/6 rounded" />
      <div className="mt-1 flex gap-2">
        <div className="skeleton h-6 w-16 rounded-full" />
        <div className="skeleton h-6 w-12 rounded-full" />
        <div className="skeleton h-6 w-20 rounded-full" />
      </div>
      <div className="flex flex-col gap-2 border-t border-slate-100 pt-3">
        {[0, 1, 2, 3].map(index => (
          <div key={index} className="flex items-center gap-3">
            <div className="skeleton h-2.5 w-20 rounded" />
            <div className="skeleton h-2 flex-1 rounded" />
            <div className="skeleton h-2.5 w-8 rounded" />
          </div>
        ))}
      </div>
    </div>
  );
}