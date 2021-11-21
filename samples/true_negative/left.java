public String toString(List<T> objList) {
  if (objList == null || objList.isEmpty()) { return ""; }
  return objList.stream().map(obj ->
    obj.toString()).collect(Collectors.joining(","));
}
