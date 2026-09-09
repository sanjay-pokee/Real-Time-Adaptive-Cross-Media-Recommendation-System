export default function SkeletonCard() {
  return (
    <div className="panel flex flex-col gap-3 p-4">
      <div className="flex gap-3.5">
        <div className="skeleton h-16 w-16 shrink-0 rounded-xl" />
        <div className="flex flex-1 flex-col gap-2 pt-1">
          <div className="skeleton h-2.5 w-20 rounded" />
          <div className="skeleton h-4 w-4/5 rounded" />
          <div className="skeleton h-2.5 w-1/2 rounded" />
        </div>
      </div>

      <div className="mt-1 flex flex-col gap-1.5">
        <div className="skeleton h-3 w-full rounded" />
        <div className="skeleton h-3 w-11/12 rounded" />
      </div>

      <div className="flex gap-1.5">
        <div className="skeleton h-5 w-16 rounded-full" />
        <div className="skeleton h-5 w-14 rounded-full" />
        <div className="skeleton h-5 w-20 rounded-full" />
      </div>

      <div className="mt-2 flex flex-col gap-2 border-t border-line pt-3.5">
        <div className="flex items-center justify-between">
          <div className="skeleton h-2.5 w-24 rounded" />
          <div className="skeleton h-3 w-10 rounded" />
        </div>
        <div className="skeleton h-2 w-full rounded-full" />
        <div className="flex gap-3">
          <div className="skeleton h-2.5 w-20 rounded" />
          <div className="skeleton h-2.5 w-16 rounded" />
        </div>
      </div>

      <div className="flex items-center gap-1 border-t border-line pt-3">
        {[0, 1, 2, 3, 4].map((index) => (
          <div key={index} className="skeleton h-7 w-7 rounded-lg" />
        ))}
        <div className="skeleton ml-auto h-3 w-20 rounded" />
      </div>
    </div>
  );
}
